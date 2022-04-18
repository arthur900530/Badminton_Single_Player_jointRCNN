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
        url1 = info[1]
        url2 = info[2]
        one_count = 0
        two_count = 0
        if name not in downloaded:
            try:
                yt = pytube.YouTube(url1)
                print(yt.streams.filter(res='1080p'))
                print('Downloading...')
                yt.streams.filter(res='1080p').first().download("../inputs/full_game_1080p")
                print('Finish download !')
                with open('downloaded.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([name, url1])
                one_count += 1
            except Exception as e:
                print(e)
                try:
                    yt = pytube.YouTube(url2)
                    print(yt.streams.filter(res='1080p'))
                    print('Downloading...')
                    yt.streams.filter(res='1080p').first().download("../inputs/full_game_1080p")
                    print('Finish download !')
                    with open('downloaded.csv', 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([name, url2])
                    two_count += 1
                except Exception as e:
                    print(e)
                    continue
        else:
            continue
    return True
