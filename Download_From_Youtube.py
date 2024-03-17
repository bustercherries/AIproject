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
    'https://www.youtube.com/watch?v=CS_WfdNij80',
    'https://www.youtube.com/watch?v=ftDgSC04zA4',
    'https://www.youtube.com/watch?v=3IpXOuoh26Q',
    'https://www.youtube.com/watch?v=fsgqP6eKugU',
    'https://www.youtube.com/watch?v=hbbNCJHEomg',
    'https://www.youtube.com/watch?v=ks1BUdskI2E',
    'https://www.youtube.com/watch?v=m6b6X-Mp5nQ'
]

output_folder = 'Filmy_Z_Youtube/wilki_test'

download_videos_from_list(urls_to_download, output_folder)