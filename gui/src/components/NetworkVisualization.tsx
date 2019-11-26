import React from 'react';
import {
  makeRequest,
  parseNetworkLayout,
  isTensorInfo,
  CANVAS_WIDTH_RANGE,
  CANVAS_MIN_WIDTH,
  CANVAS_HEIGHT_RANGE,
  CANVAS_MIN_HEIGHT,
  isConvolutionalLayerInfo,
  isActivationLayerInfo,
} from '../utils';
import TensorVisualization from './TensorVisualization';
import ActivationVisualization from './ActivationVisualization';
import Conv2dVisualization from './Conv2dVisualization';

interface NetworkVisualizationProps {
  imageData?: string;
  onClassified: (output: string) => void;
}

interface NetworkVisualizationState {
  visInfo: VisualizationInfo[];
}

export const makeVisualization = (layers: VisualizationInfo[]) => {
  let maxWidth = Number.MIN_VALUE,
    minWidth = Number.MAX_VALUE;
  let maxChannel = Number.MIN_VALUE,
    minChannel = Number.MAX_VALUE;

  for (const layer in layers) {
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

  let idx = 0;
  return layers.map(layer => {
    if (isTensorInfo(layer)) {
      return (
        <TensorVisualization
          info={layer}
          width={scaleWidth(layer.width)}
          height={scaleHeight(layer.channel)}
        />
      );
    } else if (isConvolutionalLayerInfo(layer)) {
      if (layer.type === 'Conv2d') {
        return <Conv2dVisualization info={layer} width={0} idx={idx++} />;
      } else if (layer.type === 'MaxPool2d') {
      } else {
        throw new Error('Unexpected layer: ' + layer);
      }
      return 0;
    } else if (isActivationLayerInfo(layer)) {
      return <ActivationVisualization info={layer} idx={idx++} />;
    } else {
      throw new Error('Unexpected layer: ' + layer);
    }
  });
};

class NetworkVisualization extends React.Component<
  NetworkVisualizationProps,
  NetworkVisualizationState
> {
  state: NetworkVisualizationState = {
    visInfo: [],
  };

  componentDidUpdate = (
    prevProps: NetworkVisualizationProps,
    _prevState: NetworkVisualizationState
  ) => {
    if (
      prevProps.imageData !== this.props.imageData &&
      this.props.imageData !== undefined
    ) {
      this.classify(this.props.imageData);
    }
  };

  classify = async (data: string) => {
    const callback = (context: string) => context;
    const init = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ data }),
    };
    const output = await makeRequest('classify', callback, init);
    await makeRequest('network-layout', (context: string) =>
      this.setState({
        visInfo: parseNetworkLayout(context),
      })
    );
    this.props.onClassified(output);
  };

  render() {
    return (
      <div className="NetworkArea">
        <div></div>
      </div>
    );
  }
}

export default NetworkVisualization;
