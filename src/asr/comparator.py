def compare_texts(original: str, transcribed: str):
    """
    يقارن النصين ويرجع الكلمات المختلفة
    """

    orig_words = original.split()
    trans_words = transcribed.split()

    diffs = []

    for i, word in enumerate(orig_words):
        if i >= len(trans_words):
            diffs.append(word)
            continue

        if word != trans_words[i]:
            diffs.append({
                "expected": word,
                "heard": trans_words[i]
            })

    return diffs