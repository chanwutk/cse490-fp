const SERVER_URL = 'http://127.0.0.1:5000/';

type FlowerType = 'Daisy' | 'Dandelion' | 'Rose' | 'Sunflower' | 'Tulip';
const isFlowerType = (x): x is FlowerType => {
  return (
    x === 'Daisy' ||
    x === 'Dandelion' ||
    x === 'Rose' ||
    x === 'Sunflower' ||
    x === 'Tulip'
  );
};

interface ILayer {
  readonly type: string;
  readonly inputDim: number;
  readonly outputDim: number;
  readonly outputSizeRatio: number;
}

class Conv2dLayer implements ILayer {
  readonly kernelSize: number;
  readonly stride: number;

  constructor(inputDim, outputDim, kernelSize, stride) {
    this.type = 'Conv2d';
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.kernelSize = kernelSize;
    this.stride = stride;

    this.outputSizeRatio = 1.0 / stride;
  }
}

class MaxPool2dLayer implements ILayer {
  readonly kernelSize: number;
  readonly stride: number;

  constructor(kernelSize, stride) {
    this.type = 'MaxPool2d';
    this.inputDim = 1;
    this.outputDim = 1;
    this.kernelSize = kernelSize;
    this.stride = stride;

    this.outputSizeRatio = 1.0 / stride;
  }
}

class ActivationLayer implements ILayer {
  constructor(type: string) {
    this.type = type;
    this.inputDim = 1;
    this.outputDim = 1;
    this.outputSizeRatio = 1;
  }
}
