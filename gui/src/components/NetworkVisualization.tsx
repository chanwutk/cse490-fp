import React from 'react';
import { isArray } from 'util';

interface INetworkVisualizationProps {
  imageData?: string;
  onClassified: (flowerType: FlowerType) => void;
}

interface INetworkVisualizationState {}

const makeRequest = async (
  option: string,
  callback: (context: string) => void,
  init?: RequestInit | undefined
) => {
  try {
    const response = await fetch(`${SERVER_URL}${option}`, init);

    if (!response.ok) {
      alert('Problem communicating to the server: ' + response.status);
      return;
    }

    const context = await response.text();
    callback(context);
  } catch (e) {
    alert('Unexpected error: ' + e);
  }
};

class NetworkVisualization extends React.Component<
  INetworkVisualizationProps,
  INetworkVisualizationState
> {
  state: INetworkVisualizationState = {};

  componentDidUpdate = (
    prevProps: INetworkVisualizationProps,
    _prevState: INetworkVisualizationState
  ) => {
    if (
      prevProps.imageData !== this.props.imageData &&
      this.props.imageData !== undefined
    ) {
      this.classify(this.props.imageData);
    }
  };

  classify = async (data: string) => {
    const callback = (context: string) => {
      if (isFlowerType(context)) {
        this.props.onClassified(context);
      } else {
        alert('Unexpected response from the server: ' + context);
      }
    };
    const init = {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ data }),
    };
    makeRequest('classify', callback, init);

    makeRequest('network-layout', (context: string) => {
      const parsed = JSON.parse(context);
      if (isArray(parsed)) {
        for (const { type, info } of parsed) {
          if (type === 'Conv2d') {
            const { inputDim, outputDim, kernelSize, stride } = info;
          } else if (type === 'Maxpool2d') {
            const { kernelSize, stride } = info;
          } else {
          }
        }
      }
    });
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
