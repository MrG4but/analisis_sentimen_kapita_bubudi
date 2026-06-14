Yang Dibutuhkan :
- Akun Instagram 

Cara menjalankan :

1. Masuk ke file .env dan isi IG_USER, IG_PASSWORD dengan akun Instagram anda sendiri
2. Run file analisis_sentimen_bubudi.py
3. Kode akan menjalankan Analisis Sentimen Artikel terlebih dahulu
5. Setelah selesai, kode akan menjalankan Analisis Sentimen Instagram
6. Setelah login Instagram, selesaikan Captcha/2FA terlebih dahulu jika ada
7. Setelah menyelesaikan Captcha/2FA dan masuk ke halaman utama, tekan enter di terminal agar lanjut ke postingan yang dituju
8. Tunggu kode mengambil semua komentar sekitar 15 menit (Jangan minimize tab Instagram Reelsnya)
9. Jika semua jumlah komentar sudah ditampilkan di terminal, tekan Enter pada terminal agar lanjut ke Gap Analysis
10. Jika berhasil, maka akan timbul grafik bar chart perbandingan sentimen pada layar dan Gap Analysis pada terminal
11. Tutup layer grafik bar chart lalu tunggu sekitar 5 menit agar seluruh CSV terbuat

Files yang dihasilkan :
1. Top_Words.csv
2. Sentiment_Instagram.csv
3. Sentiment_Artikel.csv
4. komparatif_sentimen.png
