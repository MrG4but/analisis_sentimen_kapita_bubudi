import os, time, re, pandas as pd
import torch
from bs4 import BeautifulSoup
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

        driver.get("https://www.instagram.com/reels/DZKW5iphoK6/")
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
            while True:
                driver.execute_script(
                    "arguments[0].scrollTop = arguments[0].scrollHeight", scroll_div
                )
                time.sleep(2)
                new_height = driver.execute_script(
                    "return arguments[0].scrollHeight", scroll_div
                )
                if new_height == last_height:
                    break
                last_height = new_height

        except Exception as e:
            print(f"Scroll div tidak ditemukan, lanjut tanpa scroll: {e}")

        soup = BeautifulSoup(driver.page_source, "html.parser")
        raw_texts = [span.get_text() for span in soup.find_all("span")]

        for text in raw_texts:

            cleaned = clean_comment(text)

            if not cleaned:
                continue

            if cleaned.lower() in seen:
                continue

            seen.add(cleaned.lower())
            comments.append(cleaned)

            print(f"Comment: {cleaned}")

            print(
                f"\nTotal comments collected: {len(comments)}"
            )

    finally:
        input(
            "\nTekan ENTER untuk menutup browser..."
        )

        driver.quit()

    return comments

comments = scrape_instagram_comments()

id_results = []

print("Starting sentiment analysis...")

for txt in comments:
    try:
        res_id = id_model(txt)[0]
        sent_id = label_map.get(res_id["label"], res_id["label"].lower())
        conf_id = round(res_id["score"], 4)

        id_results.append({"original_comment": txt, "sentiment_id": sent_id, "confidence_id": conf_id})

    except Exception as e:
        print(f"⚠️ Gagal memproses: '{txt}' | Error: {e}")

df = pd.DataFrame(id_results)
df.to_csv("comments_en_sentiment.csv", index=False, encoding="utf-8-sig")

print("\n🎉 Pipeline selesai.")
print("File saved: comments_id_sentiment.csv")