interface LayerInfo {
  readonly type: string;
}

interface ConvolutionLayerInfo extends LayerInfo {
  readonly kernelSize: number;
  readonly inputDim: number;
  readonly outputDim: number;
  readonly stride: number;
}

interface MaxPool2dLayerInfo extends LayerInfo {
  readonly kernelSize: number;
  readonly stride: number;
}

interface ActivationLayerInfo extends LayerInfo {
  readonly info?: any;
}

interface TensorInfo {
  readonly width: number;
  readonly height: number;
  readonly channel: number;
}

type VisualizationInfo = LayerInfo | TensorInfo;
