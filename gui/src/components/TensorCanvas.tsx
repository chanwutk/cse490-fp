import React from 'react';

interface TensorCanvasProps {
  reference: React.RefObject<HTMLCanvasElement>;
  size: number;
  margin: number;
}

const TensorCanvas: React.FC<TensorCanvasProps> = ({
  reference,
  size,
  margin,
}) => {
  return (
    <canvas ref={reference} width={size} height={size} style={{ margin }} />
  );
};

export default TensorCanvas;
