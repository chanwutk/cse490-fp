import React from 'react';
import { makeRequest } from '../utils';

interface MaxPool2dVisualizationProps {
  info: MaxPool2dLayerInfo;
  idx: number;
}

interface MaxPool2dVisualizationState {
  isReady: boolean;
  input?: any;
  output?: any;
}

class MaxPool2dVisualization extends React.Component<
  MaxPool2dVisualizationProps,
  MaxPool2dVisualizationState
> {
  state: MaxPool2dVisualizationState = { isReady: true };
  canvas: React.RefObject<HTMLCanvasElement>;

  constructor(props: MaxPool2dVisualizationProps) {
    super(props);
    this.canvas = React.createRef();
  }

  onClick = async () => {
    if (!this.state.isReady) {
      // TODO: request to server all the info
      const result = await makeRequest(`trace/${this.props.idx}`, JSON.parse);
      console.log(result);
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

export default MaxPool2dVisualization;
