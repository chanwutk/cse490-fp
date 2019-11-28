import React from 'react';
import './App.css';
import NetworkVisualization from './components/NetworkVisualization';
import UploadButton from './components/UploadButton';
import { CANVAS_MAX_WIDTH, makeRequest, parseNetworkLayout } from './utils';

const rootStyle: React.CSSProperties = {
  width: '100%',
  position: 'absolute',
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
};

const bodyStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  width: CANVAS_MAX_WIDTH + 'px',
};

interface AppState {
  imageData?: string;
  output: null | string;
  isUploadButtonActive: boolean;
  visualizationInfo: VisualizationInfo[];
}

class App extends React.Component<{}, AppState> {
  state: AppState = {
    output: null,
    isUploadButtonActive: true,
    visualizationInfo: [],
  };

  canvasRef: React.RefObject<HTMLCanvasElement> = React.createRef();

  handleFileUpload = (event: any) => {
    const file = event.target.files[0];
    const src = URL.createObjectURL(file);
    const image = new Image();
    const canvas = this.canvasRef.current!;
    image.src = src;
    image.onload = () => {
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(image, 0, 0, 224, 224);
      const imageData = canvas.toDataURL('image/jpeg');
      if (imageData !== this.state.imageData) {
        this.classify(imageData);
      }
    };
  };

  classify = async (data: string) => {
    const callback = (context: string) => context;
    const init = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ data }),
    };
    const output = await makeRequest('classify', callback, init);
    const visualizationInfo = await makeRequest(
      'network-layout',
      parseNetworkLayout
    );

    this.setState({
      visualizationInfo,
      output,
      isUploadButtonActive: true,
    });
  };

  render() {
    return (
      <div style={rootStyle}>
        <div style={bodyStyle}>
          {/* <h1 style={h1Style}>Rose is red; violet is blue;<br/>Which parts activate your ReLU</h1> */}
          <h1 style={{ fontSize: '40px' }}>What makes a flower its kind?</h1>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <canvas
              width="224px"
              height="224px"
              ref={this.canvasRef}
              style={{ borderStyle: 'dotted', padding: '2px' }}
            />
          </div>
          <div
            style={{
              display: 'flex',
              justifyContent: 'center',
              paddingTop: '30px',
              paddingBottom: '30px',
            }}
          >
            <UploadButton
              onClick={this.handleFileUpload}
              isActive={this.state.isUploadButtonActive}
            />
          </div>
          <NetworkVisualization
            visualizationInfo={this.state.visualizationInfo}
          />
          <div>{this.state.output ? this.state.output : ''}</div>
        </div>
      </div>
    );
  }
}

export default App;
