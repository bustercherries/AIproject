from pytube import YouTube

def download_video(url, output_folder):
    try:
        # Create a YouTube object
        yt = YouTube(url)

        # Select the first stream (usually the highest quality)
        stream = yt.streams.first()

        # Download and save the video to the specified output folder
        stream.download(output_path=output_folder)
        print(f"Video from {url} downloaded successfully!")
    except Exception as e:
        print(f"An error occurred while downloading the video from {url}: {e}")

def download_videos_from_list(urls, output_folder):
    for url in urls:
        download_video(url, output_folder)

# Lista URL do pobrania
urls_to_download = [
    'https://www.youtube.com/watch?v=OlcNU-Pyf6Y',
    'https://www.youtube.com/watch?v=h5oBwMb-Q5w',
    'https://www.youtube.com/watch?v=WnGRCFbvLCs',
    'https://www.youtube.com/watch?v=5EQAJ705py8',
    'https://www.youtube.com/watch?v=slfTpkf89pw',
    'https://www.youtube.com/watch?v=lI_ZxHd1LzM',
    'https://www.youtube.com/watch?v=qlHJZyIE7Ok'
]

output_folder = 'Filmy_Z_Youtube/wilki'

download_videos_from_list(urls_to_download, output_folder)