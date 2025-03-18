
import yt_dlp
import os

def download(dir_path, save_name, url):
    outtmpl = os.path.join(dir_path, f"{save_name}.%(ext)s")
    ydl_opts = {
        'outtmpl': outtmpl,
        'format': 'bestvideo[ext=mp4]',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])