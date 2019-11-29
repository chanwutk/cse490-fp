import React from 'react';
import {
  isTensorInfo,
  isConvolutionLayerInfo,
  isActivationLayerInfo,
  getLayersScalers,
  isMaxPoolLayerInfo,
} from '../utils';
import TensorVisualization from './TensorVisualization';
import LayerVisualization from './LayerVisualization';

interface NetworkVisualizationProps {
  visualizationInfo: VisualizationInfo[];
  imageData?: string;
}

export const makeVisualization = (
  layers: VisualizationInfo[],
  imageData?: string
) => {
  const { scaleWidth, scaleHeight } = getLayersScalers(layers);

  let idx = 0;
  return layers.map((layer: VisualizationInfo, key: number) => {
    if (isTensorInfo(layer)) {
      return (
        <TensorVisualization
          key={key}
          info={layer}
          width={scaleWidth(layer.width)}
          height={scaleHeight(layer.channel)}
        />
      );
    } else if (
      isConvolutionLayerInfo(layer) ||
      isMaxPoolLayerInfo(layer) ||
      isActivationLayerInfo(layer)
    ) {
      return (
        <LayerVisualization
          key={key}
          info={layer}
          idx={idx++}
          imageData={imageData}
        />
      );
    } else {
      throw new Error('Unexpected layer: ' + layer);
    }
  });
};

const NetworkVisualization: React.FC<NetworkVisualizationProps> = ({
  visualizationInfo,
  imageData,
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
      {makeVisualization(visualizationInfo, imageData)}
    </div>
  );
};

export default NetworkVisualization;
