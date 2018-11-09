from utils import dates


def extract_ppn(sentence, collector, avoidance):
    """
    Extract proper nouns from sentences.
    :param list[str] sentence: Sentence contains words.
    :param set collector: Detected proper nouns will be put here.
    :param set avoidance: Words in this set shouldn't recognized as
    proper nouns.
    :return:
    """
    def repeat(new_w):
        """
        Whether a proper nouns is repeated.
        :param new_w:
        :return: True if it is repeated.
        """
        parts = new_w.split()
        n = len(parts)
        for i in range(0, n):
            for j in range(i+1, n+1):
                part = ' '.join(parts[i:j])
                if part in dates or part in avoidance:
                    return True
        return False

    curr = ''
    sentence = sentence + ['end']
    # Skip the first word.
    for w in sentence[1:]:
        if w[0].isupper():
            # Try to detect continuous words
            curr += ' ' + w
        else:
            if curr == '':
                continue
            curr = curr[1:]
            if (curr not in collector) and (not repeat(curr)) and (not curr.isupper()):
                collector.add(curr)
            curr = ''
