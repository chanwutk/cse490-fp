import React from 'react';
import { makeRequest } from '../utils';

interface ActivationVisualizationProps {
  info: ActivationLayerInfo;
  idx: number;
}

interface ActivationVisualizationState {
  isReady: boolean;
  input?: any;
  weight?: any;
  output?: any;
}

class ActivationVisualization extends React.Component<
  ActivationVisualizationProps,
  ActivationVisualizationState
> {
  state: ActivationVisualizationState = { isReady: true };
  canvas: React.RefObject<HTMLCanvasElement>;

  constructor(props: ActivationVisualizationProps) {
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

export default ActivationVisualization;
