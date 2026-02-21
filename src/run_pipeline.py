from humanizer.text_humanizer import humanize
from performance.performance_layer import performance_enhance, strip_markers_for_tts

if __name__ == "__main__":
    sample = "الذكاء الاصطناعي غير طريقة صناعة المحتوى بالكامل. هذا التطور سريع جدا."
    h = humanize(sample)
    p = performance_enhance(h)

    print("\n--- Humanized ---")
    print(h)

    print("\n--- Performance (with markers) ---")
    print(p)

    print("\n--- Ready for TTS (markers stripped) ---")
    print(strip_markers_for_tts(p))