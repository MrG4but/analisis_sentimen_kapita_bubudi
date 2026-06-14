import os, time, re, pandas as pd
import torch
import requests
import matplotlib.pyplot as plt
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from bs4 import BeautifulSoup
from collections import Counter
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN1")
UsernameIG = os.getenv("IG_USER")
PasswordIG = os.getenv("IG_PASSWORD")

device = 0 if torch.cuda.is_available() else -1
id_model = pipeline("text-classification", model="crypter70/IndoBERT-Sentiment-Analysis", token=HF_TOKEN, device=device)
label_map = {"LABEL_0": "negative", "LABEL_1": "positive"}

FILTER_UI = [

    "View all", "reply", "see translation", "likes", "like", "follow",
    "following", "original audio", "suggested for you", "more posts from",
    "log in", "sign up", "Meta", "threads", "Down chevron icon", "Meta in Indonesia",
    "days ago"
]

Stopword_tambahan = {
    'nya', 'aja', 'udah', 'udh', 'gak', 'ga', 'gua', 'gue', 'lo', 'lu',
    'sih', 'deh', 'dong', 'lah', 'nih', 'tuh', 'yg', 'yuk', 'wkwk',
    'pak', 'bro', 'sis', 'sama', 'juga', 'emang', 'kayak', 'kaya',
    'banget', 'bgt', 'mau', 'bisa', 'kita', 'itu', 'ini', 'ada', 'dan'
}

factory = StopWordRemoverFactory()
Stop_Words = set(factory.get_stop_words()) | Stopword_tambahan

url_artikel = "https://money.kompas.com/read/2026/06/04/113502926/dollar-hari-ini-tembus-rp-18000-ini-penyebab-rupiah-terpuruk-ke-level-terendah?page=all"
post_url = "https://www.instagram.com/reels/DZKW5iphoK6/"

def clean_comment(comment):

    comment = re.sub(r"http\S+", "", comment)
    comment = re.sub(r"@\w+", "", comment)
    comment = re.sub(r"@[\w.]+", "", comment)

    lower_comment = comment.lower()

    if lower_comment in [ui.lower() for ui in FILTER_UI]:
        return None

    if "·" in comment:
        return None

    if len(comment.split()) < 3:
        return None

    if not re.search(r"[a-zA-Z]", comment):
        return None

    if re.search(r"view all \d+ repl", lower_comment):
        return None

    if re.search(r"and \d+ others", lower_comment):
        return None

    return comment

def open_comments(driver):
    wait = WebDriverWait(driver, 10)

    try:
        comment_btn = wait.until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "svg[aria-label='Comment']")
        ))
        comment_btn.click()
        time.sleep(2)
        return True
    except Exception as e:
        pass

def is_username(teks):
    return bool(re.match(r'^[\w.]+$', teks))

def fetch_article_text(url):
    response = requests.get(url)
    print(f"Status: {response.status_code}")

    soup = BeautifulSoup(response.text, 'html.parser')

    clearfix_divs = soup.find_all('div', class_='clearfix')
    print(f"Total div.clearfix: {len(clearfix_divs)}")

    article_div = max(clearfix_divs, key=lambda d: len(d.find_all('p')))

    paragraphs = article_div.find_all('p')
    skip_starts = ['baca juga', 'artikel ini', 'simak breaking news', 'dapatkan update' 'freepik']

    texts = []
    for p in paragraphs:
        teks = p.get_text().strip()
        if len(teks) < 50:
            continue
        if any(teks.lower().startswith(s) for s in skip_starts):  # ← tambahan
            continue
        texts.append(teks)

    print(f"Total paragraf ditemukan: {len(texts)}")
    return texts

def scrape_instagram_comments(post_url: str) -> list:
    driver = webdriver.Chrome()

    comments = []
    seen = set()

    try:
        driver.get("https://www.instagram.com/accounts/login/")

        time.sleep(5)

        username_input = driver.find_element(By.NAME,"email")
        password_input = driver.find_element(By.NAME,"pass")

        username_input.send_keys(UsernameIG)
        time.sleep(2)
        password_input.send_keys(PasswordIG)
        time.sleep(1)
        password_input.send_keys(Keys.RETURN)


        time.sleep(8)

        input(
            "Jika CAPTCHA muncul, selesaikan lalu tekan ENTER..."
        )

        driver.get(post_url)
        time.sleep(5)
        wait = WebDriverWait(driver, 10)

        try:
            comment_btn = wait.until(EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "svg[aria-label='Comment']")
            ))
            comment_btn.click()
            time.sleep(10)

            scroll_div = driver.find_element(
                By.XPATH, "//div[@style and contains(@style, 'minHeight')]"
            )
            last_height = driver.execute_script(
                "return arguments[0].scrollHeight", scroll_div
            )

            no_change_count = 0

            while True:
                driver.execute_script(
                    "arguments[0].scrollTop = arguments[0].scrollHeight", scroll_div
                )
                time.sleep(2)
                new_height = driver.execute_script(
                    "return arguments[0].scrollHeight", scroll_div
                )

                if new_height == last_height:
                    no_change_count += 1
                    if no_change_count >= 3:  # fungsinya kalo scrollnya udah 3x tapi nda berubah, ini bakalan stopin scrolnya
                        break
                else:
                    no_change_count = 0

                last_height = new_height
                print(f"Scrolling... height: {new_height}")  # hapus ja ni nanti

        except Exception as e:
            print(f"Scroll div tidak ditemukan, lanjut tanpa scroll: {e}")

        dialog = driver.find_element(By.XPATH, "//div[@role='dialog']")
        all_spans = dialog.find_elements(By.XPATH, ".//span[@dir='auto']")

        skip_keywords = ['Comments', 'Reply', 'View all', 'likes', 'follow', 'Like' 'See translation']

        for span in all_spans:
            teks = span.text.strip()

            if not teks:
                continue
            if len(teks) < 5:
                continue
            if any(kw.lower() in teks.lower() for kw in skip_keywords):
                continue

            if is_username(teks):
                continue

            if re.match(r'^\d+[wdhm]', teks):
                continue

            if teks.lower() in seen:
                continue

            cleaned = clean_comment(teks)
            if not cleaned:
                continue

            seen.add(teks.lower())
            comments.append(cleaned)
            print(f"Comment: {cleaned}")

        print(f"\nTotal comments collected: {len(comments)}")

    finally:
        input(
            "\nTekan ENTER untuk menutup browser..."
        )

        driver.quit()

    return comments

def analisis_sentimen_artikel(paragraphs: list) -> pd.DataFrame:
    results = []

    for sentence in paragraphs:
        try:
            sentence = sentence[:500]
            res = id_model(sentence)[0]
            sentiment = label_map.get(res["label"], res["label"].lower())
            confidence = round(res["score"], 4)

            results.append({
                "sentence": sentence,
                "sentiment": sentiment,
                "confidence": confidence
            })
            print(f"[{sentiment}] {sentence[:60]}...")

        except Exception as e:
            print(f"⚠️ Gagal: {e}")

    df_artikel = pd.DataFrame(results)
    df_artikel.to_csv("artikel_sentiment.csv", index=False, encoding="utf-8-sig")
    print("\nFile saved: artikel_sentiment.csv")

    counts = df_artikel['sentiment'].value_counts()
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index, counts.values, color=['green', 'red'])
    plt.xlabel('Sentimen')
    plt.ylabel('Jumlah Kalimat')
    plt.title('Distribusi Sentimen Artikel')
    plt.tight_layout()
    plt.show()

    return df_artikel

def analisis_sentimen_ig(comments:list) -> pd.DataFrame:
    id_results = []

    print("Starting sentiment analysis...")

    for txt in comments:
        try:
            res_id = id_model(txt)[0]
            sent_id = label_map.get(res_id["label"], res_id["label"].lower())
            conf_id = round(res_id["score"], 4)

            id_results.append({"text": txt, "sentiment": sent_id, "confidence": conf_id})

        except Exception as e:
            print(f"Gagal memproses: '{txt}' | Error: {e}")

    df_ig = pd.DataFrame(id_results)
    df_ig.to_csv("comments_id_sentiment.csv", index=False, encoding="utf-8-sig")
    print("File saved: comments_id_sentiment.csv")
    return df_ig

def gap_analysis(df_artikel: pd.DataFrame, df_ig: pd.DataFrame):
    print("\nGap Analysis: Media vs Publik...")

    all_artikel_words = []
    for txt in df_artikel["sentence"]:
        words = [w for w in txt.lower().split()
                 if w not in Stop_Words and w.isalpha() and len(w) > 3]
        all_artikel_words.extend(words)

    artikel_freq = Counter(all_artikel_words)
    media_words  = set([w for w, _ in artikel_freq.most_common(20)])

    # Keyword dari komentar IG
    all_ig_words = []
    for txt in df_ig["text"]:
        words = [w for w in txt.lower().split()
                 if w not in Stop_Words and w.isalpha() and len(w) > 3]
        all_ig_words.extend(words)

    ig_freq      = Counter(all_ig_words)
    publik_words = set([w for w, _ in ig_freq.most_common(20)])

    only_media   = media_words - publik_words
    only_publik  = publik_words - media_words

    # Sentimen publik
    sentiment_dist = df_ig["sentiment"].value_counts(normalize=True).mul(100).round(1)
    print(f"\n   Sentimen publik (IG): {sentiment_dist.to_dict()}")
    print(f"   Keyword HANYA di media  : {list(only_media)[:10]}")
    print(f"   Keyword HANYA di publik : {list(only_publik)[:10]}")

    df_neg = df_ig[df_ig["sentiment"] == "negative"]
    df_pos = df_ig[df_ig["sentiment"] == "positive"]

    # Keyword di komentar negatif
    neg_words = []
    for txt in df_neg["text"]:
        neg_words.extend([w for w in txt.lower().split()
                          if w not in Stop_Words and w.isalpha() and len(w) > 3])
    print("\nTop keyword di komentar NEGATIF:")
    print(Counter(neg_words).most_common(15))

    pos_words = []
    for txt in df_pos["text"]:
        pos_words.extend([w for w in txt.lower().split()
                          if w not in Stop_Words and w.isalpha() and len(w) > 3])
    print("\nTop keyword di komentar POSITIF:")
    print(Counter(pos_words).most_common(15))

    # ✅ Bar chart komparatif artikel vs IG — untuk slide PPT
    artikel_sentiment = df_artikel["sentiment"].value_counts()
    ig_sentiment      = df_ig["sentiment"].value_counts()

    labels   = ["negative", "positive"]
    artikel_vals = [artikel_sentiment.get(l, 0) for l in labels]
    ig_vals      = [ig_sentiment.get(l, 0) for l in labels]

    x     = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width/2 for i in x], artikel_vals, width, label="Artikel Berita", color="#4C72B0")
    ax.bar([i + width/2 for i in x], ig_vals,      width, label="Komentar IG",    color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Jumlah")
    ax.set_title("Perbandingan Sentimen: Artikel Berita vs Komentar Instagram")
    ax.legend()
    plt.tight_layout()
    plt.savefig("komparatif_sentimen.png", dpi=150)
    plt.show()

    # Simpan gap analysis
    df_gap = pd.DataFrame({
        "only_in_media":  pd.Series(list(only_media)[:15]),
        "only_in_public": pd.Series(list(only_publik)[:15])
    })
    df_gap.to_csv("gap_analysis.csv", index=False, encoding="utf-8-sig")
    print("File Saved: gap_analysis.csv")

    return sentiment_dist, only_media, only_publik

if __name__ == "__main__":
    # 1. Artikel berita → keyword + graph
    artikel = fetch_article_text(url_artikel)
    df_artikel = analisis_sentimen_artikel(artikel)

    # 2. Instagram → scrape + sentiment
    komentar = scrape_instagram_comments(post_url)
    df_ig    = analisis_sentimen_ig(komentar)

    # 3. Gap analysis
    sentiment_dist, only_media, only_publik = gap_analysis(df_artikel, df_ig)

    print("\n" + "="*55)
    print("PIPELINE SELESAI. File yang dihasilkan:")
    print("artikel_keywords.csv")
    print("instagram_sentimen.csv")
    print("gap_analysis.csv")
    print("="*55)