import cv2
import os
import pickle



SEQUENCES = [
    '00',
    '01',
    '02',
    '03',
    '04',
    '05',
    '06',
    '07',
    '08',
    '08',
    '10',
]


def main():
    cherry_picked_frames = []
    for sequence in SEQUENCES:
        indir = 'data/SEMANTIC-KITTI-DATASET/sequences/' + sequence + '/'
        imgdir = indir + 'image_2/'
        veldir = indir + 'velodyne/'
        print('\n#############')
        print('Sequence Number: ', sequence)
        for frame in range(len(os.listdir(imgdir))):
            imgfile = imgdir + str(frame).zfill(6) + '.png'
            img = cv2.imread(imgfile)
            cv2.imshow('frame', img)
            key = cv2.waitKey(0)
            if key & 0xFF == ord('s'):
                # Pick this data point
                cherry_picked_frames.append([sequence, str(frame)])
                print('Saved images: ', len(cherry_picked_frames))

            if key & 0xFF == ord('n'):
                # Next sequence
                break
            if key & 0xFF == ord('q'):
                # Quit
                break

        if key & 0xFF == ord('q'):
                break

    with open('cherry_picked_frames.pkl', 'wb') as f: pickle.dump(cherry_picked_frames, f)
    breakpoint()



if __name__=='__main__':
    main()