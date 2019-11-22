import React from 'react';
import './App.css';
import CSS from 'csstype';
import { FileInput } from 'react-md';
import NetworkVisualization from './components/NetworkVisualization';

const rootStyle: CSS.Properties = {
  height: '100%',
  width: '100%',
  position: 'absolute',
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
};

const bodyStyle: CSS.Properties = {
  height: '100%',
  minWidth: '800px',
  display: 'flex',
  // backgroundColor: 'black',  // TODO: will change to whtie
  flexDirection: 'column',
};

const h1Style: CSS.Properties = {
  fontSize: '40px'
}

interface IAppProps {};

interface IAppState {
  imagePath?: string;
  output?: FlowerType;
};

class App extends React.Component<IAppProps, IAppState> {

  updateImagePath = (imagePath: string) => {
    this.setState({imagePath});
  }

  test = (a: any) => {
    console.log(a.target.files);
  }

  render() {
    return (
      <div style={rootStyle}>
        <div style={bodyStyle}>
          <h1 style={h1Style}>What makes a flower its kind?</h1>
          <FileInput
            id="file-input-image"
            accept="image/*"
            name="images"
            onChange={this.test}
           >Upload Image</FileInput>
          {/* <input
            accept="image/*"
            style={{ display: 'none' }}
            id="icon-button-photo"
            onChange={this.test}
            type="file"
          />
          <label htmlFor="icon-button-photo">
            <Button color="primary" component="span">
              Upload Image
            </Button>
          </label>
          <Button onClick={() => {}}>
            Upload Image
            <input
              type='file'
              style={{ display: "none" }}
            />
           </Button> */}
          <NetworkVisualization/>
          <div>{() => {
            if (this.state.output !== null) {
              return '';
            } else {
              return this.state.output;
            }
          }}</div>
        </div>
      </div>
    );
  }
}

export default App;
