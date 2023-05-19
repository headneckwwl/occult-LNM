
from _algo.utils.ImageSpider import download_image

if __name__ == '__main__':
    download_image('labels.txt', engines=['sogou'], save_to='train', ask_num=100, num_process=4)
    download_image('labels.txt', engines=['bing'], save_to='val', ask_num=10, num_process=1)
