import React from 'react';
import { makeRequest } from '../utils';

interface Conv2dVisualizationProps {
  info: ConvolutionalLayerInfo;
  width: number;
  idx: number;
}

interface Conv2dVisualizationState {
  isReady: boolean;
  input?: any;
  weight?: any;
  output?: any;
}

class Conv2dVisualization extends React.Component<
  Conv2dVisualizationProps,
  Conv2dVisualizationState
> {
  state: Conv2dVisualizationState = { isReady: true };
  canvas: React.RefObject<HTMLCanvasElement>;

  constructor(props: Conv2dVisualizationProps) {
    super(props);
    this.canvas = React.createRef();
  }

  onClick = async () => {
    if (!this.state.isReady) {
      // TODO: request to server all the info
      await makeRequest(`trace/${this.props.idx}`, (context: string) => {
        // TODO: parse context to the representation of the tensor
      });
      this.setState({
        isReady: true,
      });
    }
    // display all the info
  };

  render() {
    return (
      <div>
        <canvas ref={this.canvas} onClick={this.onClick}></canvas>
      </div>
    );
  }
}

export default Conv2dVisualization;
