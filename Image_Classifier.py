from sklearn import svm
from sklearn.metrics import zero_one_loss
from sklearn.cross_validation import train_test_split
from scipy import io
import numpy
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#####Problem 4
spamTrainingData = io.loadmat('data/spam-dataset/spam_data.mat')
spamSamples = spamTrainingData['training_data']
spamLabels = numpy.ravel(spamTrainingData['training_labels'])
while True:
	spamTrainingSamples = []
	spamTrainingLabels = []
	spamValidationSamples = []
	spamValidationLabels = []
	randomIndex = random.sample(range(0,len(spamSamples)), len(spamSamples))
	for i in randomIndex[:4000]:
		spamTrainingSamples.append(spamSamples[i])
		spamTrainingLabels.append(spamLabels[i])
	for i in randomIndex[4000:]:
		spamValidationSamples.append(spamSamples[i])
		spamValidationLabels.append(spamLabels[i])
	spamTrainingSamples = numpy.array(spamTrainingSamples)
	spamTrainingLabels = numpy.array(spamTrainingLabels)
	spamValidationSamples = numpy.array(spamValidationSamples)
	spamValidationLabels = numpy.array(spamValidationLabels)
	if (len(numpy.unique(spamTrainingLabels)) == 2):
		break

print 'starting'
spamClassifierArray = [svm.SVC(C = 100.5, kernel = 'linear'),svm.SVC(C = 100.4, kernel = 'linear'),
	svm.SVC(C = 100.3, kernel = 'linear'),svm.SVC(C = 100.2, kernel = 'linear'),
	svm.SVC(C = 100.1, kernel = 'linear'),svm.SVC(C = 100, kernel = 'linear'),
	svm.SVC(C = .001, kernel = 'linear'),svm.SVC(C = .0001, kernel = 'linear'),
	svm.SVC(C = .0001, kernel = 'linear'),svm.SVC(C = .0001, kernel = 'linear'),]
spamErrorArray = [0,0,0,0,0,0,0,0,0,0]

for j in range(0,10):
	spamFoldTrainingSamples = numpy.concatenate((spamTrainingSamples[0:400*j],spamTrainingSamples[400*(j+1):4000]))
	spamFoldTrainingLabels = numpy.concatenate((spamTrainingLabels[0:400*j],spamTrainingLabels[400*(j+1):4000]))
	spamFoldValidationSamples = spamTrainingSamples[400*j:400*(j+1)]
	spamFoldValidationLabels = spamTrainingLabels[400*j:400*(j+1)]
	for k in range(0,6):
		spamClassifierArray[k].fit(spamFoldTrainingSamples, spamFoldTrainingLabels)
		spamPrediction = spamClassifierArray[k].predict(spamFoldValidationSamples)
		spamErrorArray[k] += zero_one_loss(spamFoldValidationLabels,spamPrediction)
	print 'ran'
print spamErrorArray
