import React from 'react';
import './App.css';
import NetworkVisualization from './components/NetworkVisualization';
import UploadButton from './components/UploadButton';
import {
  CANVAS_MAX_WIDTH,
  makeRequest,
  parseNetworkLayout,
  layerThinDivStyle,
  drawArrow,
  layerRootStyle,
} from './utils';

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
  justifyContent: 'center',
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

  pictureRef: React.RefObject<HTMLCanvasElement> = React.createRef();
  lastArrowRef: React.RefObject<HTMLCanvasElement> = React.createRef();

  handleFileUpload = (event: any) => {
    const file = event.target.files[0];
    const src = URL.createObjectURL(file);
    const image = new Image();
    const canvas = this.pictureRef.current!;
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

    await this.setState({
      visualizationInfo,
      output,
      isUploadButtonActive: true,
      imageData: data,
    });

    drawArrow(this.lastArrowRef.current!);
  };

  render() {
    return (
      <div style={rootStyle}>
        <div style={bodyStyle}>
          <div style={{ marginBottom: 40 }}>
            <h1 style={{ fontSize: '40px', marginBottom: 0 }}>
              What makes a flower its kind?
            </h1>
            <p style={{ marginTop: 5 }}>
              Rose is red; violet is blue; Which pixels activate your ReLU
            </p>
          </div>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <canvas
              width="224px"
              height="224px"
              ref={this.pictureRef}
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
            imageData={this.state.imageData}
          />
          {this.state.output ? (
            <div style={layerRootStyle}>
              <div
                style={{
                  justifyContent: 'right',
                  ...layerThinDivStyle,
                }}
              >
                <div style={{ marginRight: '10px' }}>Classification Layers</div>
              </div>
              <canvas ref={this.lastArrowRef} width="80" height="80" />
              <div style={layerThinDivStyle} />
            </div>
          ) : null}
          <div
            style={{
              display: 'flex',
              justifyContent: 'center',
              fontSize: 50,
              marginBottom: 250,
            }}
          >
            {this.state.output ? `This picture is ${this.state.output}!` : ''}
          </div>
        </div>
      </div>
    );
  }
}

export default App;
