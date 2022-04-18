import pytube
import csv


def get_vid_paths(path):
    vid_infos = []
    with open(path, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            vid_infos.append(row)
    return vid_infos


def download_vid(vid_infos):
    downloaded = []
    with open('downloaded.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            if len(row) != 0:
                downloaded.append(row[0])
            else:
                continue
    for info in vid_infos:
        name = info[0]
        url = info[1]
        if name not in downloaded:
            yt = pytube.YouTube(url)
            print(yt.streams.filter(res='1080p'))
            print('Downloading...')
            yt.streams.filter(res='1080p').first().download("../inputs/full_game_1080p")
            print('Finish download !')
            with open('downloaded.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([name, url])
        else:
            continue
    return True
