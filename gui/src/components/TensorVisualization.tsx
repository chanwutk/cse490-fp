import React from 'react';

interface TensorVisualizationProps {
  info: TensorInfo;
  width: number;
  height: number;
}

const TensorVisualization: React.FC<TensorVisualizationProps> = ({
  info,
  width,
  height,
}) => {
  return (
    <div
      style={{ width, height, textAlign: 'center' }}
    >{`${info.width} x ${info.height} x ${info.channel}`}</div>
  );
};

export default TensorVisualization;
