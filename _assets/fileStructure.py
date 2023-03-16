import os

input_dir = 'D:/Somani/D/mito_doubleTag/Deconvolved'
output_dir = 'D:/Somani/D/Output'
vid_list = os.listdir(input_dir)

def main():
	for n in vid_list:
		img_fold = os.path.join(output_dir, n[0:-4])
		if not os.path.isdir(img_fold):
			os.makedirs(img_fold)
			os.makedirs(os.path.join(img_fold,'bri'))
			os.makedirs(os.path.join(img_fold,'flu1'))
			os.makedirs(os.path.join(img_fold,'flu2'))

if __name__ == '__main__':
    main()