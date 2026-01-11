from __future__ import division
import random
import nltk
from nltk.stem.wordnet import WordNetLemmatizer

# =========================
# Optional backend: pattern / pattern3 / lemminflect
# =========================
_HAS_PATTERN = False
_HAS_PATTERN3 = False
_HAS_LEMMINFLECT = False

# try pattern.en
try:
    from pattern.en import conjugate, PRESENT, SG, PL, PAST, PROGRESSIVE
    _HAS_PATTERN = True
except Exception:
    _HAS_PATTERN = False

# try pattern3.en (often works when pattern.en doesn't)
if not _HAS_PATTERN:
    try:
        from pattern3.en import conjugate, PRESENT, SG, PL, PAST, PROGRESSIVE
        _HAS_PATTERN3 = True
    except Exception:
        _HAS_PATTERN3 = False

# try lemminflect as a modern fallback
if (not _HAS_PATTERN) and (not _HAS_PATTERN3):
    try:
        import lemminflect
        _HAS_LEMMINFLECT = True
    except Exception:
        _HAS_LEMMINFLECT = False


# =========================
# Function 0: stop words
# =========================
def get_stopwords():
    stop_words = [
        'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain',
        'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'am', 'among',
        'amongst', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway',
        'anywhere', 'are', 'aren', "aren't", 'around', 'as', 'at', 'back', 'been', 'before',
        'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond',
        'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
        "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either',
        'else', 'elsewhere', 'empty', 'enough', 'even', 'ever', 'everyone', 'everything',
        'everywhere', 'except', 'first', 'for', 'former', 'formerly', 'from', 'hadn',
        "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence', 'her', 'here',
        'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself',
        'his', 'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn',
        "isn't", 'it', "it's", 'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll',
        'may', 'me', 'meanwhile', 'mightn', "mightn't", 'mine', 'more', 'moreover', 'most',
        'mostly', 'must', 'mustn', "mustn't", 'my', 'myself', 'namely', 'needn', "needn't",
        'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none', 'noone', 'nor',
        'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
        'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out',
        'over', 'per', 'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've",
        'shouldn', "shouldn't", 'somehow', 'something', 'sometime', 'somewhere', 'such', 't',
        'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then',
        'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon',
        'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to',
        'too', 'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've',
        'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'whatever',
        'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein',
        'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever',
        'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won', "won't",
        'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're",
        "you've", 'your', 'yours', 'yourself', 'yourselves'
    ]
    return set(stop_words)


UniversalPos = ['NOUN', 'VERB', 'ADJ', 'ADV',
                'PRON', 'DET', 'ADP', 'NUM',
                'CONJ', 'PRT', '.', 'X']


# =========================
# Function 1: POS tagging
# =========================
def _ensure_nltk_pos_tagger():
    """
    只做轻量检查：如果资源缺失，给出明确提示，避免用户卡在奇怪的 LookupError。
    """
    try:
        nltk.pos_tag(["test"])
    except LookupError as e:
        raise LookupError(
            "NLTK 资源缺失：你需要下载 POS tagger。\n"
            "在 python 里执行：\n"
            "  import nltk\n"
            "  nltk.download('averaged_perceptron_tagger')\n"
            "  nltk.download('universal_tagset')\n"
            f"原始错误：{e}"
        )


def get_pos(sent, tagset='universal'):
    """
    :param sent: list[str]
    :param tagset: {'universal', 'default'}
    :return: tuple/list of pos tags
    """
    _ensure_nltk_pos_tagger()

    if tagset == 'default':
        word_n_pos_list = nltk.pos_tag(sent)
    elif tagset == 'universal':
        word_n_pos_list = nltk.pos_tag(sent, tagset=tagset)
    else:
        raise ValueError("tagset must be 'universal' or 'default'")

    _, pos_list = zip(*word_n_pos_list)
    return pos_list


# =========================
# Function 2: POS Filter
# =========================
def pos_filter(ori_pos, new_pos_list):
    same = [
        True if (ori_pos == new_pos or (set([ori_pos, new_pos]) <= set(['NOUN', 'VERB'])))
        else False
        for new_pos in new_pos_list
    ]
    return same


# =========================
# Function 3: Verb tense detection
# =========================
def get_v_tense(sent):
    """
    :param sent: list[str]
    :return: dict {index: PennTreebank tense tag (VB/VBD/VBG/VBN/VBP/VBZ)}
    """
    _ensure_nltk_pos_tagger()
    word_n_pos_list = nltk.pos_tag(sent)
    _, pos_list = zip(*word_n_pos_list)
    tenses = {w_ix: tense for w_ix, tense in enumerate(pos_list) if tense.startswith('V')}
    return tenses


# =========================
# Function 4: Change tense
# =========================
def _change_tense_with_pattern(word, tense):
    """
    用 pattern / pattern3 做动词变形。
    """
    lookup = {
        'VB':  conjugate(verb=word, tense=PRESENT, number=SG),
        'VBD': conjugate(verb=word, tense=PAST, aspect=PROGRESSIVE, number=SG),
        'VBG': conjugate(verb=word, tense=PRESENT, aspect=PROGRESSIVE, number=SG),
        'VBN': conjugate(verb=word, tense=PAST, aspect=PROGRESSIVE, number=SG),
        'VBP': conjugate(verb=word, tense=PRESENT, number=PL),
        'VBZ': conjugate(verb=word, tense=PRESENT, number=SG),
    }
    out = lookup.get(tense, None)
    return out if out else word


def _change_tense_with_lemminflect(word, tense):
    """
    用 lemminflect 做动词变形：更现代，也更稳定。
    """
    # lemminflect 的 tag 使用 Penn Treebank tag（VB/VBD/VBG/VBN/VBP/VBZ）正好匹配
    infl = lemminflect.getInflection(word, tag=tense)
    if infl and len(infl) > 0:
        return infl[0]
    return word


def change_tense(word, tense, lemmatize=False):
    """
    将 word 变为指定时态（tense: VB/VBD/VBG/VBN/VBP/VBZ）。
    若 backend 不可用，则返回原词（保证不崩）。
    """
    if lemmatize:
        word = WordNetLemmatizer().lemmatize(word, 'v')

    if _HAS_PATTERN or _HAS_PATTERN3:
        return _change_tense_with_pattern(word, tense)
    if _HAS_LEMMINFLECT:
        return _change_tense_with_lemminflect(word, tense)

    # 最差降级：不做变形，避免整个攻击流程崩
    return word


# =========================
# Debug helpers (optional)
# =========================
def get_sent_list():
    file_format = "/afs/csail.mit.edu/u/z/zhijing/proj/to_di/data/{}/test_lm.txt"
    content = []
    for dataset in ['ag', 'fake', 'mr', 'yelp']:
        file = file_format.format(dataset)
        with open(file) as f:
            content += [line.strip().split() for line in f if line.strip()]
    return content


def check_pos(sent_list, win_size=10):
    sent_list = sent_list[:]
    random.shuffle(sent_list)
    sent_list = sent_list[:100]

    center_ix = [
        random.randint(0 + win_size // 2, len(sent) - 1 - win_size // 2) if len(sent) > win_size else len(sent) // 2
        for sent in sent_list
    ]
    word_range = [[max(0, cen_ix - win_size // 2), min(len(sent), cen_ix + win_size // 2)]
                  for cen_ix, sent in zip(center_ix, sent_list)]

    corr_pos = [get_pos(sent)[word_range[sent_ix][0]: word_range[sent_ix][1]] for sent_ix, sent in enumerate(sent_list)]
    part_pos = [get_pos(sent[word_range[sent_ix][0]: word_range[sent_ix][1]]) for sent_ix, sent in enumerate(sent_list)]

    diff_s_ix = []
    for sent_ix, (sent_pos_corr, sent_pos_part) in enumerate(zip(corr_pos, part_pos)):
        cen_ix = center_ix[sent_ix] - word_range[sent_ix][0]
        if sent_pos_corr[cen_ix] != sent_pos_part[cen_ix]:
            diff_s_ix += [sent_ix]

    if diff_s_ix:
        import pdb
        pdb.set_trace()


def main():
    stop_words = get_stopwords()

    sent = 'i have a dream'.split()
    pos_list = get_pos(sent)
    tenses = get_v_tense(sent)

    # 这句应该不再因为 pattern.en 导入问题而崩
    new_word = change_tense('made', 'VBD')
    print("change_tense('made','VBD') =", new_word)


if __name__ == "__main__":
    main()
