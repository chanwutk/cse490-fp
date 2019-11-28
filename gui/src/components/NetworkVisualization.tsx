import React from 'react';
import {
  isTensorInfo,
  isConvolutionLayerInfo,
  isActivationLayerInfo,
  getLayersScalers,
  isMaxPoolLayerInfo,
} from '../utils';
import TensorVisualization from './TensorVisualization';
import ActivationVisualization from './ActivationVisualization';
import LayerVisualization from './LayerVisualization';
import MaxPool2dVisualization from './MaxPool2dVisualization';

interface NetworkVisualizationProps {
  visualizationInfo: VisualizationInfo[];
}

export const makeVisualization = (layers: VisualizationInfo[]) => {
  const { scaleWidth, scaleHeight } = getLayersScalers(layers);

  let idx = 0;
  layers = layers.map((layer: VisualizationInfo, key: number) => {
    if (isTensorInfo(layer)) {
      return (
        <TensorVisualization
          key={key}
          info={layer}
          width={scaleWidth(layer.width)}
          height={scaleHeight(layer.channel)}
        />
      );
    } else if (isConvolutionLayerInfo(layer)) {
      return <LayerVisualization key={key} info={layer} idx={idx++} />;
    } else if (isMaxPoolLayerInfo(layer)) {
      return <MaxPool2dVisualization key={key} info={layer} idx={idx++} />;
    } else if (isActivationLayerInfo(layer)) {
      return <ActivationVisualization key={key} info={layer} idx={idx++} />;
    } else {
      throw new Error('Unexpected layer: ' + layer);
    }
  });
  return layers;
};

const NetworkVisualization: React.FC<NetworkVisualizationProps> = ({
  visualizationInfo,
}) => {
  return (
    <div
      className="NetworkArea"
      style={{
        borderStyle: 'dot',
        borderColor: 'black',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}
    >
      {makeVisualization(visualizationInfo)}
    </div>
  );
};

export default NetworkVisualization;
