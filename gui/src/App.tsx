import React from 'react';
import './App.css';
import CSS from 'csstype';
import NetworkVisualization from './components/NetworkVisualization';
import UploadButton from './components/UploadButton';
import { CANVAS_MAX_WIDTH } from './utils';

const rootStyle: CSS.Properties = {
  width: '100%',
  position: 'absolute',
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
};

const bodyStyle: CSS.Properties = {
  minWidth: CANVAS_MAX_WIDTH + 'px',
  display: 'flex',
  flexDirection: 'column',
};

const h1Style: CSS.Properties = {
  fontSize: '40px',
};

interface AppProps {}

interface AppState {
  imageData?: string;
  output: null | string;
  isUploadButtonActive: boolean;
}

class App extends React.Component<AppProps, AppState> {
  state: AppState = {
    output: null,
    isUploadButtonActive: true,
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
        this.setState({
          imageData: canvas.toDataURL('image/jpeg'),
          isUploadButtonActive: false,
        });
      }
    };
  };

  handleClassified = (output: string) => {
    this.setState({
      output,
      isUploadButtonActive: true,
    });
  };

  render() {
    return (
      <div style={rootStyle}>
        <div style={bodyStyle}>
          {/* <h1 style={h1Style}>Rose is red; violet is blue;<br/>Which parts activate your ReLU</h1> */}
          <h1 style={h1Style}>What makes a flower its kind?</h1>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <canvas
              width="224px"
              height="224px"
              ref={this.canvasRef}
              style={{ borderStyle: 'dotted' }}
            />
          </div>
          <div
            style={{
              display: 'flex',
              justifyContent: 'center',
              paddingTop: '30px',
            }}
          >
            <UploadButton
              onClick={this.handleFileUpload}
              isActive={this.state.isUploadButtonActive}
            />
          </div>
          <NetworkVisualization
            imageData={this.state.imageData}
            onClassified={this.handleClassified}
          />
          <div>{this.state.output ? this.state.output : ''}</div>
        </div>
      </div>
    );
  }
}

export default App;
