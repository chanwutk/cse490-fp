import React from 'react';

interface INetworkVisualizationProps {
  imageData?: string;
  onClassified: (flowerType: FlowerType) => void;
}

interface INetworkVisualizationState {
  imageData?: string;
}

class NetworkVisualization extends React.Component<
  INetworkVisualizationProps,
  INetworkVisualizationState
> {
  state: INetworkVisualizationState = {};

  componentDidUpdate = () => {
    if (this.state.imageData !== this.props.imageData) {
      this.setState({
        imageData: this.props.imageData,
      });
    }
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
