import React from 'react';
import './App.css';
import CSS from 'csstype';
import NetworkVisualization from './components/NetworkVisualization';
import UploadButton from './components/UploadButton';

const rootStyle: CSS.Properties = {
  width: '100%',
  position: 'absolute',
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
};

const bodyStyle: CSS.Properties = {
  minWidth: '800px',
  display: 'flex',
  flexDirection: 'column',
};

const h1Style: CSS.Properties = {
  fontSize: '40px',
};

interface IAppProps {}

interface IAppState {
  imageData?: string;
  output: null | FlowerType;
  isUploadButtonActive: boolean;
}

class App extends React.Component<IAppProps, IAppState> {
  state: IAppState = {
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
      ctx.drawImage(image, 0, 0, 500, 500);
      this.setState({
        imageData: canvas.toDataURL(),
        isUploadButtonActive: false,
      });
    };
  };

  handleClassified = (flowerType: FlowerType) => {
    this.setState({
      output: flowerType,
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
              width="500px"
              height="500px"
              ref={this.canvasRef}
              style={{ width: 500, height: 500 }}
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
