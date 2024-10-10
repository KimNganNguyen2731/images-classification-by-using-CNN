import matplotlib.pyplot as plt


def loss_curve(loss_result):
  plt.plot(loss_result, color = 'red')
  plt.xlabel('Step')
  plt.ylabel('Loss')
  plt.title('Loss Curve')
  plt.savefig('loss_curve.png')
  #plt.show()
