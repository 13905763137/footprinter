from datagen import generator
import matplotlib.pyplot as plt
plt.switch_backend("Agg") 
import tensorflow as tf

def main(BS):

  train, val = generator(BS)
  img, mask = val.__next__()
  for i in range(BS):
    print(img[i].min(), img[i].max())
    print(mask[i].min(), mask[i].max())
    fig, ax = plt.subplots(nrows = 1, ncols = 2)
    ax[0].imshow(img[i])
    ax[1].imshow(mask[i,:,:,0])
    plt.savefig("/home/vakili/public_html/building/"+str(i)+".png")
    plt.close()

  return None

BS = 10
main(BS)
