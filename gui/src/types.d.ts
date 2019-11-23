import { thisTypeAnnotation } from '@babel/types';

type FlowerType = 'Daisy' | 'Dandelion' | 'Rose' | 'Sunflower' | 'Tulip';

interface ILayer {
  type: string;
}

class Conv2dLayer implements ILayer {
  name: string;
  inputDim: number;
  outputDim: number;
  kernelSize: number;

  constructor(
    name: string,
    inputDim: number,
    outputDim: number,
    kernelSize: number
  ) {
    this.name = name;
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.kernelSize = kernelSize;
  }
}
