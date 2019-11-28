import { isArray } from 'util';

export const SERVER_PORT = 5432;
export const SERVER_URL = `http://127.0.0.1:${SERVER_PORT}/`;
export const CANVAS_MAX_WIDTH = 800;
export const CANVAS_MIN_WIDTH = 120;
export const CANVAS_WIDTH_RANGE = CANVAS_MAX_WIDTH - CANVAS_MIN_WIDTH;
export const CANVAS_MAX_HEIGHT = 300;
export const CANVAS_MIN_HEIGHT = 50;
export const CANVAS_HEIGHT_RANGE = CANVAS_MAX_HEIGHT - CANVAS_MIN_HEIGHT;

export const makeTensorInfo = (
  width: number,
  height: number,
  channel: number
): TensorInfo => {
  return { width, height, channel };
};
export const INITIAL_TENSOR = makeTensorInfo(224, 224, 3);

export const isLayerInfo = (x: any): x is LayerInfo => {
  return x.type !== undefined;
};

export const isConvolutionLayerInfo = (x: any): x is ConvolutionLayerInfo => {
  if (x.type && x.type === 'Conv2d') {
    return (
      x.inputDim !== undefined &&
      x.outputDim !== undefined &&
      x.stride !== undefined &&
      x.kernelSize !== undefined
    );
  }
  return false;
};
export const isMaxPoolLayerInfo = (x: any): x is MaxPool2dLayerInfo => {
  if (x.type && x.type === 'MaxPool2d') {
    return x.stride !== undefined && x.kernelSize !== undefined;
  }
  return false;
};

export const isActivationLayerInfo = (x: any): x is ActivationLayerInfo => {
  return x.type && (x.type !== 'Conv2d' || x.type !== 'MaxPool2d');
};

export const isTensorInfo = (x: any): x is TensorInfo => {
  return (
    x.width !== undefined && x.height !== undefined && x.channel !== undefined
  );
};

export const makeRequest = async <T>(
  option: string,
  callback: (context: string) => T,
  init?: RequestInit | undefined
): Promise<T> => {
  // try {
  console.log(`${SERVER_URL}${option}`);
  const response = await fetch(`${SERVER_URL}${option}`, init);

  if (!response.ok) {
    throw new Error('Problem communicating to the server: ' + response.status);
  }

  const context = await response.text();
  return callback(context);
  // } catch (e) {
  //   throw new Error('Unexpected error: ' + e);
  // }
};

export const parseNetworkLayout = (text: string) => {
  const parsed = JSON.parse(text);
  if (isArray(parsed)) {
    let prevTensorInfo = INITIAL_TENSOR;
    const layout: VisualizationInfo[] = [INITIAL_TENSOR];
    for (const layer of parsed) {
      let nextTensorInfo: TensorInfo;

      if (isConvolutionLayerInfo(layer)) {
        const nextWidth = ~~(prevTensorInfo.width / layer.stride);
        const nextHeight = ~~(prevTensorInfo.height / layer.stride);
        if (prevTensorInfo.channel !== layer.inputDim) {
          throw new Error(
            `Layers dimension missmatch (previous tensor: ${prevTensorInfo}, layer: ${layer})`
          );
        }

        nextTensorInfo = makeTensorInfo(nextWidth, nextHeight, layer.outputDim);
      } else if (isMaxPoolLayerInfo(layer)) {
        const nextWidth = ~~(prevTensorInfo.width / layer.stride);
        const nextHeight = ~~(prevTensorInfo.height / layer.stride);

        nextTensorInfo = makeTensorInfo(
          nextWidth,
          nextHeight,
          prevTensorInfo.channel
        );
      } else if (isActivationLayerInfo(layer)) {
        nextTensorInfo = layout[layout.length - 1] as TensorInfo;
      } else {
        throw new Error('Unexpected layer type from the server' + layer);
      }

      layout.push(...[layer, nextTensorInfo]);
      prevTensorInfo = nextTensorInfo;
    }

    return layout;
  } else {
    throw new Error('parsed context should be an array');
  }
};

export const getBufferFromBytes = (imageBytes: number[]): Buffer => {
  const imageBuffer = new Buffer(imageBytes.length);
  for (let b = 0; b < imageBytes.length; b++) {
    imageBuffer[b] = imageBytes[b];
  }
  return imageBuffer;
};

export const getLayersScalers = (layers: VisualizationInfo[]) => {
  let maxWidth = Number.MIN_VALUE,
    minWidth = Number.MAX_VALUE;
  let maxChannel = Number.MIN_VALUE,
    minChannel = Number.MAX_VALUE;

  for (const layer of layers) {
    if (isTensorInfo(layer)) {
      const { width, channel } = layer;
      maxWidth = Math.max(width, maxWidth);
      minWidth = Math.min(width, minWidth);
      maxChannel = Math.max(channel, maxChannel);
      minChannel = Math.min(channel, minChannel);
    }
  }

  const widthScale = CANVAS_WIDTH_RANGE / (maxWidth - minWidth);
  const scaleWidth = (width: number) =>
    (width - minWidth) * widthScale + CANVAS_MIN_WIDTH;

  const heightScale = CANVAS_HEIGHT_RANGE / (maxChannel - minChannel);
  const scaleHeight = (channel: number) =>
    (channel - minChannel) * heightScale + CANVAS_MIN_HEIGHT;

  return {
    scaleWidth,
    scaleHeight,
  };
};
