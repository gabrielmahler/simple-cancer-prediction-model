import torch
from models.dl_model import DLModel
import numpy as np

def loadDLModel():
    model = DLModel()
    model.load_state_dict(torch.load('models/dl_model.pth'))
    return model


def main():
    models = dict()
    models['dl'] = loadDLModel()

    while True:
        print('DL model loaded')
        age = int(input('Enter age: '))
        gender = int(input('Enter gender (0 for male, 1 for female): '))
        bmi = float(input('Enter BMI: '))
        smoking = int(input('Enter smoking status (0 for non-smoker, 1 for smoker): '))
        genetic_risk = int(input('Enter genetic risk (0 indicating Low, 1 indicating Medium, and 2 indicating High): '))
        physical_activity = int(input('Enter physical activity (0 - 10): '))
        alcohol_intake = int(input('Enter alcohol intake (0 - 5): '))
        input_data = np.array([age, gender, bmi, smoking, genetic_risk, physical_activity, alcohol_intake])
        input_data = torch.tensor(input_data, dtype=torch.float32)
        output = models['dl'](input_data)

        print(f'Diagnosis prediction: {output[0]}')


if __name__ == '__main__':
    main()