import os, time, re, pandas as pd
import torch
import requests
import matplotlib.pyplot as plt
import networkx as nx
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

post_url = "https://www.instagram.com/reels/DZKW5iphoK6/"
url_artikel = "https://money.kompas.com/read/2026/06/04/113502926/dollar-hari-ini-tembus-rp-18000-ini-penyebab-rupiah-terpuruk-ke-level-terendah?page=all"



def scrapping_artikel(url:str):
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')

    paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
    article_text = ' '.join(paragraphs)

    # Keyword extraction
    words = article_text.split()
    filtered_words = [w for w in words if w.lower() not in Stop_Words and w.isalpha()]
    word_freq = Counter(filtered_words)

    # Hanya kata yang muncul minimal 2x — filter noise
    final_words = [w for w in filtered_words if word_freq[w] >= 2]
    top_keywords = word_freq.most_common(20)

    print(f"   Top 10 keyword artikel: {top_keywords[:10]}")

    # Simpan keyword ke CSV
    df_keywords = pd.DataFrame(top_keywords, columns=["keyword", "freq_artikel"])
    df_keywords.to_csv("artikel_keywords.csv", index=False, encoding="utf-8-sig")
    print("File saved: artikel_keywords.csv")

    # Co-occurrence graph — untuk slide PPT
    G = nx.Graph()
    for i in range(len(final_words) - 1):
        w1, w2 = final_words[i], final_words[i + 1]
        if G.has_edge(w1, w2):
            G[w1][w2]['weight'] += 1
        else:
            G.add_edge(w1, w2, weight=1)

    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    max_w = max(weights) if weights else 1
    normalized_widths = [1 + (w / max_w) * 3 for w in weights]

    plt.figure(figsize=(14, 14))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color="#7B68EE")
    nx.draw_networkx_edges(G, pos, width=normalized_widths, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.axis('off')
    plt.title('Jaringan Kata — Artikel Berita Dolar Rp18.000', fontsize=14)
    plt.tight_layout()
    plt.savefig("artikel_cooccurrence_graph.png", dpi=150)
    plt.show()

    return df_keywords, final_words



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

def analisis_sentimen(comments:list) -> pd.DataFrame:
    id_results = []

    print("Starting sentiment analysis...")

    for txt in comments:
        try:
            res_id = id_model(txt)[0]
            sent_id = label_map.get(res_id["label"], res_id["label"].lower())
            conf_id = round(res_id["score"], 4)

            id_results.append({"text": txt, "sentiment": sent_id, "confidence": conf_id})

        except Exception as e:
            print(f"⚠️ Gagal memproses: '{txt}' | Error: {e}")

    df_ig = pd.DataFrame(id_results)
    df_ig.to_csv("comments_id_sentiment.csv", index=False, encoding="utf-8-sig")
    print("File saved: comments_id_sentiment.csv")
    return df_ig

def gap_analysis(df_keywords: pd.DataFrame, df_ig: pd.DataFrame):
    print("\nGap Analysis: Media vs Publik...")

    # Keyword dari komentar IG
    all_ig_words = []
    for txt in df_ig["text"]:
        words = [w for w in txt.lower().split()
                 if w not in Stop_Words and w.isalpha() and len(w) > 2]
        all_ig_words.extend(words)

    ig_freq     = Counter(all_ig_words)
    top_ig      = dict(ig_freq.most_common(20))
    media_words = set(df_keywords["keyword"].str.lower().tolist())
    publik_words = set(top_ig.keys())

    only_media  = media_words - publik_words   # Media bahas, publik tidak
    only_publik = publik_words - media_words   # Publik resah, media tidak liput

    # Sentimen publik
    sentiment_dist = df_ig["sentiment"].value_counts(normalize=True).mul(100).round(1)

    print(f"\n   Sentimen publik (IG):")
    print(f"   {sentiment_dist.to_dict()}")
    print(f"\n   Keyword HANYA di media  : {list(only_media)[:10]}")
    print(f"   Keyword HANYA di publik : {list(only_publik)[:10]}")

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
    df_keywords, _ = scrapping_artikel(url_artikel)

    # 2. Instagram → scrape + sentiment
    komentar = scrape_instagram_comments(post_url)
    df_ig    = analisis_sentimen(komentar)

    # 3. Gap analysis
    sentiment_dist, only_media, only_publik = gap_analysis(df_keywords, df_ig)

    print("\n" + "="*55)
    print("PIPELINE SELESAI. File yang dihasilkan:")
    print("artikel_keywords.csv")
    print("artikel_cooccurrence_graph.png")
    print("instagram_sentimen.csv")
    print("gap_analysis.csv")
    print("="*55)