import argparse
import os
import random
import numpy as np

import dataloader
from train_classifier import Model
import criteria

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModel

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class USE_PT(object):
    def __init__(self, model_name_or_path="microsoft/Multilingual-MiniLM-L12-H384", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name_or_path, torch_dtype="auto").to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _encode(self, sents, batch_size=64, max_length=256):
        embs = []
        for i in range(0, len(sents), batch_size):
            batch = sents[i:i+batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)

            # mean pooling with attention mask
            last_hidden = out.last_hidden_state  # [B, L, H]
            mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, L, 1]
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

            pooled = F.normalize(pooled, p=2, dim=1)
            embs.append(pooled)

        return torch.cat(embs, dim=0)  # [N, H]

    def semantic_sim(self, sents1, sents2, batch_size=64):
        assert len(sents1) == len(sents2)
        emb1 = self._encode(sents1, batch_size=batch_size)
        emb2 = self._encode(sents2, batch_size=batch_size)

        cos = (emb1 * emb2).sum(dim=1).clamp(-1.0, 1.0)
        sim = 1.0 - torch.acos(cos)
        return [sim.detach().cpu().numpy()]


def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    sim_words, sim_values = [], []
    for i, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[i]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[i][mask], sim_value[mask]
        sim_word = [idx2word[j] for j in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values


class HFSequenceClassifier(nn.Module):
    def __init__(self, model_name_or_path, nclasses, max_seq_length=128, batch_size=32, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        # tokenizer + model
        # Load model directly
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=2,
            torch_dtype="auto"  # PyTorch里用 torch_dtype，而不是 dtype
        ).to(self.device)

    def _encode_batch(self, texts):
        """
        texts: List[List[str]]  (你原来传进来的 text_data 每个元素是 token list)
        返回 dict[str, Tensor]，可直接喂给 transformers 模型
        """
        # 兼容你原来的输入：List[token] -> str
        sents = [" ".join(t) for t in texts]

        enc = self.tokenizer(
            sents,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )

        # 某些模型（如 RoBERTa/DistilBERT）没有 token_type_ids，这里不强行补
        return enc

    def text_pred(self, text_data, batch_size=32):
        self.model.eval()

        probs_all = []
        for i in range(0, len(text_data), batch_size):
            batch_texts = text_data[i:i + batch_size]
            enc = self._encode_batch(batch_texts)

            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                out = self.model(**enc)
                logits = out.logits
                probs = torch.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def attack(text_ls, true_label, predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
           batch_size=32):
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()

    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0

    len_text = len(text_ls)
    if len_text < sim_score_window:
        sim_score_threshold = 0.1
    half_sim_score_window = (sim_score_window - 1) // 2
    num_queries = 1

    pos_ls = criteria.get_pos(text_ls)

    leave_1_texts = [text_ls[:ii] + ['<oov>'] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
    leave_1_probs = predictor(leave_1_texts, batch_size=batch_size)
    num_queries += len(leave_1_texts)

    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
    import_scores = (orig_prob - leave_1_probs[:, orig_label] +
                     (leave_1_probs_argmax != orig_label).float() *
                     (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
                     ).data.cpu().numpy()

    words_perturb = []
    for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
        if score > import_score_threshold and text_ls[idx] not in stop_words_set:
            words_perturb.append((idx, text_ls[idx]))

    words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
    synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)

    synonyms_all = []
    for idx, word in words_perturb:
        if word in word2idx:
            synonyms = synonym_words.pop(0)
            if synonyms:
                synonyms_all.append((idx, synonyms))

    text_prime = text_ls[:]
    text_cache = text_prime[:]
    num_changed = 0

    device = next(predictor.__self__.model.parameters()).device if hasattr(predictor, "__self__") else get_device()

    for idx, synonyms in synonyms_all:
        new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
        new_probs = predictor(new_texts, batch_size=batch_size)

        if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
            text_range_min = idx - half_sim_score_window
            text_range_max = idx + half_sim_score_window + 1
        elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
            text_range_min = 0
            text_range_max = sim_score_window
        elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
            text_range_min = len_text - sim_score_window
            text_range_max = len_text
        else:
            text_range_min = 0
            text_range_max = len_text

        semantic_sims = sim_predictor.semantic_sim(
            [' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
            [' '.join(x[text_range_min:text_range_max]) for x in new_texts],
        )[0]

        num_queries += len(new_texts)
        if len(new_probs.shape) < 2:
            new_probs = new_probs.unsqueeze(0)

        new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
        new_probs_mask *= (semantic_sims >= sim_score_threshold)

        synonyms_pos_ls = [
            criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)] if len(new_text) > 10
            else criteria.get_pos(new_text)[idx]
            for new_text in new_texts
        ]
        pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
        new_probs_mask *= pos_mask

        if np.sum(new_probs_mask) > 0:
            text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
            num_changed += 1
            break
        else:
            penalty = torch.from_numpy(((semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float))).float().to(device)
            new_label_probs = new_probs[:, orig_label] + penalty
            new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
            if new_label_prob_min < orig_prob:
                text_prime[idx] = synonyms[new_label_prob_argmin]
                num_changed += 1

        text_cache = text_prime[:]

    return ' '.join(text_prime), num_changed, orig_label, torch.argmax(predictor([text_prime])), num_queries


def random_attack(text_ls, true_label, predictor, perturb_ratio, stop_words_set, word2idx, idx2word, cos_sim,
                  sim_predictor=None, import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15,
                  synonym_num=50, batch_size=32):
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()

    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0

    len_text = len(text_ls)
    if len_text < sim_score_window:
        sim_score_threshold = 0.1
    half_sim_score_window = (sim_score_window - 1) // 2
    num_queries = 1

    pos_ls = criteria.get_pos(text_ls)

    perturb_idxes = random.sample(range(len_text), max(1, int(len_text * perturb_ratio)))
    words_perturb = [(idx, text_ls[idx]) for idx in perturb_idxes]

    words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
    synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)

    synonyms_all = []
    for idx, word in words_perturb:
        if word in word2idx:
            synonyms = synonym_words.pop(0)
            if synonyms:
                synonyms_all.append((idx, synonyms))

    text_prime = text_ls[:]
    text_cache = text_prime[:]
    num_changed = 0

    device = next(predictor.__self__.model.parameters()).device if hasattr(predictor, "__self__") else get_device()

    for idx, synonyms in synonyms_all:
        new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
        new_probs = predictor(new_texts, batch_size=batch_size)

        if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
            text_range_min = idx - half_sim_score_window
            text_range_max = idx + half_sim_score_window + 1
        elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
            text_range_min = 0
            text_range_max = sim_score_window
        elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
            text_range_min = len_text - sim_score_window
            text_range_max = len_text
        else:
            text_range_min = 0
            text_range_max = len_text

        semantic_sims = sim_predictor.semantic_sim(
            [' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
            [' '.join(x[text_range_min:text_range_max]) for x in new_texts],
        )[0]

        num_queries += len(new_texts)
        if len(new_probs.shape) < 2:
            new_probs = new_probs.unsqueeze(0)

        new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
        new_probs_mask *= (semantic_sims >= sim_score_threshold)

        synonyms_pos_ls = [
            criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)] if len(new_text) > 10
            else criteria.get_pos(new_text)[idx]
            for new_text in new_texts
        ]
        pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
        new_probs_mask *= pos_mask

        if np.sum(new_probs_mask) > 0:
            text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
            num_changed += 1
            break
        else:
            penalty = torch.from_numpy(((semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float))).float().to(device)
            new_label_probs = new_probs[:, orig_label] + penalty
            new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
            if new_label_prob_min < orig_prob:
                text_prime[idx] = synonyms[new_label_prob_argmin]
                num_changed += 1

        text_cache = text_prime[:]

    return ' '.join(text_prime), num_changed, orig_label, torch.argmax(predictor([text_prime])), num_queries


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--nclasses", type=int, default=2)

    parser.add_argument("--target_model", type=str, required=True, choices=['wordLSTM', 'bert', 'wordCNN'])
    parser.add_argument("--target_model_path", type=str, required=True)

    parser.add_argument("--word_embeddings_path", type=str, default='')
    parser.add_argument("--counter_fitting_embeddings_path", type=str, required=True)
    parser.add_argument("--counter_fitting_cos_sim_path", type=str, default='')

    # 原来的 --USE_cache_path 不再需要；为了兼容旧命令行，这里改成可选且不使用
    parser.add_argument("--USE_cache_path", type=str, default=None, help="(deprecated) not used in PyTorch version.")

    # 新增：相似度模型选择
    parser.add_argument("--sim_model_name", type=str,
                        default="/mnt/shared-storage-user/liuqihua/NLP_experiment/Final/Final_Work_Liuqihua_TextFooler/models/Multilingual-MiniLM-L12-H384",
                        help="SentenceTransformer model for semantic similarity.")

    parser.add_argument("--output_dir", type=str, default='adv_results')

    parser.add_argument("--sim_score_window", default=15, type=int)
    parser.add_argument("--import_score_threshold", default=-1., type=float)
    parser.add_argument("--sim_score_threshold", default=0.7, type=float)
    parser.add_argument("--synonym_num", default=50, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--data_size", default=200, type=int)
    parser.add_argument("--perturb_ratio", default=0., type=float)
    parser.add_argument("--max_seq_length", default=128, type=int)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    texts, labels = dataloader.read_corpus(args.dataset_path)
    data = list(zip(texts, labels))[:args.data_size]
    print("Data import finished!")

    device = get_device()

    print("Building Model...")
    if args.target_model == 'wordLSTM':
        model = Model(args.word_embeddings_path, nclasses=args.nclasses).to(device)
        checkpoint = torch.load(args.target_model_path, map_location=device)
        model.load_state_dict(checkpoint)
    elif args.target_model == 'wordCNN':
        model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=100, cnn=True).to(device)
        checkpoint = torch.load(args.target_model_path, map_location=device)
        model.load_state_dict(checkpoint)
    elif args.target_model == 'bert':
        model = HFSequenceClassifier(args.target_model_path, nclasses=args.nclasses, max_seq_length=args.max_seq_length)

    predictor = model.text_pred
    print("Model built!")

    idx2word, word2idx = {}, {}
    print("Building vocab...")
    with open(args.counter_fitting_embeddings_path, 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    print("Building cos sim matrix...")
    if args.counter_fitting_cos_sim_path:
        print(f'Load pre-computed cosine similarity matrix from {args.counter_fitting_cos_sim_path}')
        cos_sim = np.load(args.counter_fitting_cos_sim_path)
    else:
        print('Start computing the cosine similarity matrix!')
        embeddings = []
        with open(args.counter_fitting_embeddings_path, 'r') as ifile:
            for line in ifile:
                embedding = [float(num) for num in line.strip().split()[1:]]
                embeddings.append(embedding)
        embeddings = np.array(embeddings)
        product = np.dot(embeddings, embeddings.T)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        cos_sim = product / np.dot(norm, norm.T)
    print("Cos sim import finished!")

    # 语义相似度模块（PyTorch）
    use = USE_PT(model_name_or_path=args.sim_model_name)

    orig_failures = 0.
    adv_failures = 0.
    changed_rates = []
    nums_queries = []
    orig_texts, adv_texts = [], []
    true_labels, new_labels = [], []

    log_file = open(os.path.join(args.output_dir, 'results_log'), 'a')
    stop_words_set = criteria.get_stopwords()

    print('Start attacking!')
    for i, (text, true_label) in enumerate(data):
        if i % 20 == 0:
            print(f'{i} samples out of {args.data_size} have been finished!')

        if args.perturb_ratio > 0.:
            new_text, num_changed, orig_label, new_label, num_queries = random_attack(
                text, true_label, predictor, args.perturb_ratio,
                stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=use,
                sim_score_threshold=args.sim_score_threshold,
                import_score_threshold=args.import_score_threshold,
                sim_score_window=args.sim_score_window,
                synonym_num=args.synonym_num,
                batch_size=args.batch_size
            )
        else:
            new_text, num_changed, orig_label, new_label, num_queries = attack(
                text, true_label, predictor,
                stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=use,
                sim_score_threshold=args.sim_score_threshold,
                import_score_threshold=args.import_score_threshold,
                sim_score_window=args.sim_score_window,
                synonym_num=args.synonym_num,
                batch_size=args.batch_size
            )

        if true_label != orig_label:
            orig_failures += 1
        else:
            nums_queries.append(num_queries)

        if true_label != new_label:
            adv_failures += 1

        changed_rate = 1.0 * num_changed / max(1, len(text))

        # if true_label == orig_label and true_label != new_label:
        changed_rates.append(changed_rate)
        orig_texts.append(' '.join(text))
        adv_texts.append(new_text)
        true_labels.append(true_label)
        new_labels.append(new_label)

    denom = float(len(data)) if len(data) > 0 else 1.0
    message = ('For target model {}: original accuracy: {:.3f}%, adv accuracy: {:.3f}%, '
               'avg changed rate: {:.3f}%, num of queries: {:.1f}\n').format(
        args.target_model,
        (1 - orig_failures / denom) * 100,
        (1 - adv_failures / denom) * 100,
        (np.mean(changed_rates) * 100) if len(changed_rates) > 0 else 0.0,
        (np.mean(nums_queries)) if len(nums_queries) > 0 else 0.0
    )
    print(message)
    log_file.write(message)
    log_file.close()

    with open(os.path.join(args.output_dir, 'adversaries.txt'), 'w', encoding='utf-8') as ofile:
        for orig_text, adv_text, tl, nl in zip(orig_texts, adv_texts, true_labels, new_labels):
            ofile.write(f'orig sent ({tl}):\t{orig_text}\nadv sent ({nl}):\t{adv_text}\n\n')


if __name__ == "__main__":
    main()
