import os
import sys
import urllib
import json
import multiprocessing
from skimage import io
import argparse
import time
import datetime


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

SORT_BY_OPTIONS = ['interestingness-desc', 'relevance', 'date-taken-desc']
DOWNLOAD_URL_KEY = 'url_l'

#API_KEY = '<API_KEY_HERE>'
API_KEY = '05101beb98cbbb128b3ecc4788386aa1'
SORT_BY = SORT_BY_OPTIONS[1]

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--query', required=True)
parser.add_argument('-s', '--save_to', required=True)
parser.add_argument('-n', '--num_requested', type=int, required=True)
parser.add_argument('-p', '--start_page', type=int, default=1)
args = parser.parse_args()

query = args.query.replace(' ', '_')

print Colors.OKGREEN + 'Searching for ' + query + ', sorted by ' + SORT_BY + Colors.ENDC
print Colors.OKGREEN + 'Downloading ' + str(args.num_requested) + Colors.ENDC

current_page = args.start_page

file_urls = set()
file_paths = []
previous_urls_obtained = 0
max_taken_date = int(time.time())
date_range = 10000000

save_to_dir = os.path.abspath(args.save_to)
if not os.path.exists(save_to_dir):
    os.makedirs(save_to_dir)

while len(file_urls) < args.num_requested:
    params = urllib.urlencode({
        'method': 'flickr.photos.search',
        'api_key': API_KEY,
        'text': query,
        'safe_search': 2,
        'content_type': 1,
        'media': 'photos',
        'per_page': 500,
        'format': 'json',
        'license': '0, 1, 2, 3, 4, 5, 6, 7, 8',
        'nojsoncallback': 1,
        'sort': SORT_BY,
        'extras': DOWNLOAD_URL_KEY,  # Retrieve download URL
        'page': current_page,
        'max_taken_date': max_taken_date,
        'min_taken_date': max_taken_date - date_range
    })

    url = 'https://api.flickr.com/services/rest/' + '?' + params
    decoded = json.loads(urllib.urlopen(url).read().decode('utf-8'))
    photos = decoded['photos']['photo']

    for photo in photos:
        if DOWNLOAD_URL_KEY in photo:
            url = photo.get(DOWNLOAD_URL_KEY)
            file_urls.add(url)
            file_paths.append(os.path.join(save_to_dir, query + str(len(file_urls)) + '.jpg'))
            sys.stdout.write("\rDiscovered %d unique images as of page %d" % (len(file_urls), current_page))
            sys.stdout.flush()

            if len(file_urls) >= args.num_requested:
                break

    if len(file_urls) == previous_urls_obtained:
        print '\n' + Colors.FAIL + 'No new images obtained in new request. Changing taken dates...' + Colors.ENDC
        max_taken_date -= date_range + 1
        print (Colors.FAIL + 'Max taken date changed to ' + str(max_taken_date) + ' (' +
            datetime.datetime.fromtimestamp(max_taken_date).strftime('%Y-%m-%d %H:%M:%S') +
            ')' + Colors.ENDC)
        current_page = 0

    previous_urls_obtained = len(file_urls)
    current_page += 1
print '\n'


def download_image(args_tuple):
    """For use with multiprocessing map. Returns filename on fail."""
    try:
        url, filename = args_tuple
        if not os.path.exists(filename):
            urllib.urlretrieve(url, filename)
        io.imread(filename)
        return True
    except KeyboardInterrupt:
        raise Exception()  # multiprocessing doesn't catch keyboard exceptions
    except:
        return False

print 'Downloading...'
num_workers = 16
if num_workers <= 0:
    num_workers = multiprocessing.cpu_count() + num_workers
pool = multiprocessing.Pool(processes=num_workers)
map_args = zip(file_urls, file_paths)
results = pool.map(download_image, map_args)
