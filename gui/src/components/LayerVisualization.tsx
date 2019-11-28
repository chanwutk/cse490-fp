import React from 'react';
import { makeRequest } from '../utils';
import { Fab } from '@material-ui/core';
import LayerTracePopup from './LayerTracePopup';

const rootStyle: React.CSSProperties = {
  padding: '5px',
  display: 'flex',
  alignItems: 'center',
  flexDirection: 'row',
};

const thinDivStyle: React.CSSProperties = {
  width: '0px',
  display: 'flex',
  alignItems: 'center',
};

interface LayerVisualizationProps {
  info: ConvolutionLayerInfo;
  idx: number;
}

interface LayerVisualizationState {
  isReady: boolean;
  isPopupOpen: boolean;
  input?: any;
  weights?: any;
  output?: any;
}

class LayerVisualization extends React.Component<
  LayerVisualizationProps,
  LayerVisualizationState
> {
  state: LayerVisualizationState = { isReady: false, isPopupOpen: false };
  canvas: React.RefObject<HTMLCanvasElement>;

  constructor(props: LayerVisualizationProps) {
    super(props);
    this.canvas = React.createRef();
  }

  componentDidMount() {
    const canvas = this.canvas.current!;
    const ctx = canvas.getContext('2d')!;
    ctx.beginPath();
    ctx.rect(20, 0, 40, 40);
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(0, 40);
    ctx.lineTo(80, 40);
    ctx.lineTo(40, 80);
    ctx.fill();
  }

  onClick = async () => {
    if (!this.state.isReady) {
      const result = await makeRequest(`trace/${this.props.idx}`, JSON.parse);
      this.setState({
        isReady: true,
        isPopupOpen: true,
        ...result,
      });
    } else {
      this.setState({
        isPopupOpen: true,
      });
    }
  };

  closePopup = () => {
    this.setState({
      isPopupOpen: false,
    });
  };

  render() {
    return (
      <div style={rootStyle}>
        <div style={{ justifyContent: 'right', ...thinDivStyle }}>
          <div style={{ marginRight: '10px' }}>{this.props.info.type}</div>
        </div>
        <canvas ref={this.canvas} width="80" height="80" />
        <div style={thinDivStyle}>
          <Fab
            variant="extended"
            color="primary"
            style={{
              left: '10px',
              paddingLeft: '65px',
              paddingRight: '65px',
            }}
            onClick={this.onClick}
          >
            Visualize!
          </Fab>
        </div>
        {this.state.isReady ? (
          <LayerTracePopup
            input={this.state.input}
            output={this.state.output}
            weights={this.state.weights}
            isOpen={this.state.isPopupOpen}
            onClose={this.closePopup}
          />
        ) : null}
      </div>
    );
  }
}

export default LayerVisualization;
