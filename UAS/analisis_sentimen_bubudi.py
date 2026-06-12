import os, time, re, pandas as pd
import torch
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from transformers import pipeline
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
UsernameIG = os.getenv("IG_USER")
PasswordIG = os.getenv("IG_PASS")

device = 0 if torch.cuda.is_available() else -1

en_model = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    token=HF_TOKEN,
    device=device
)

translator = GoogleTranslator(
    source='id',
    target='en'
)

FILTER_UI = [

    "View all", "reply", "see translation", "likes", "like", "follow",
    "following", "original audio", "suggested for you", "more posts from",
    "log in", "sign up", "Meta", "threads", "Down chevron icon", "Meta in Indonesia",
    "days ago"
]

def clean_comment(comment):

    comment = re.sub(r"http\S+", "", comment)
    comment = re.sub(r"@\w+", "", comment)
    comment = re.sub(r"#\w+", "", comment)

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

def translate_comment(comment):

    try:

        translasi = translator.translate(comment)

        print(f"Translation: {translasi}")

        return translasi

    except Exception as e:

        print(f"translation Error: {e}")

        return comment

def scrape_instagram_comments():
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

        hashtag = "indomie"
        driver.get(
            f"https://www.instagram.com/explore/tags/{hashtag}/"
        )
        time.sleep(5)

        for i in range(1):
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )

            time.sleep(4)

        post_links = set()

        posts = driver.find_elements(
            By.XPATH,
            "//a[contains(@href, '/p/')]"
        )

        for post in posts:

            href = post.get_attribute("href")

            if href:
                post_links.add(href)

        print(f"\nCollected {len(post_links)} posts")

        for index, link in enumerate(list(post_links)[:10]):

            print(f"\nOpening post {index+1}")
            print(link)

            driver.get(link)

            time.sleep(5)

            # scroll comment section
            for _ in range(2):
                driver.execute_script(
                    "window.scrollTo(0, document.body.scrollHeight);"
                )

                time.sleep(2)

            soup = BeautifulSoup(
                driver.page_source,
                "html.parser"
            )

            raw_texts = [
                span.get_text()
                for span in soup.find_all("span")]

            print(
                f"💬 Raw texts found: {len(raw_texts)}"
            )

            valid_comments = 0

            for text in raw_texts:

                cleaned = clean_comment(text)

                if not cleaned:
                    continue

                if cleaned.lower() in seen:
                    continue

                seen.add(cleaned.lower())
                comments.append(cleaned)

                valid_comments += 1

                print(f"Comment: {cleaned}")

            print(
                f"Valid comments: {valid_comments}"
            )

        print(
            f"\nTotal comments collected: {len(comments)}"
        )

        return comments

    finally:

        input(
            "\nTekan ENTER untuk menutup browser..."
        )

        driver.quit()

comments = scrape_instagram_comments()

en_results = []

print("Starting sentiment analysis...")

for txt in comments:

    try:
        time.sleep(0.4)

        translated = translate_comment(txt)
        res_en = en_model(translated)[0]

        sentiment = res_en["label"]
        nilai = round(res_en["score"], 4)

        print(
            f"Sentiment: {sentiment} "
            f"Nilai: {nilai}"
        )

        en_results.append({"original_comment": txt, "translated_comment": translated, "sentiment_en": sentiment, "confidence_en": nilai})

    except Exception as e:
        print(
            f"⚠️ Gagal memproses: '{txt}' | Error: {e}"
        )

df = pd.DataFrame(en_results)
df.to_csv("comments_en_sentiment.csv", index=False, encoding="utf-8-sig")

print("\n🎉 Pipeline selesai.")
print("File saved: comments_en_sentiment.csv")

assdasd