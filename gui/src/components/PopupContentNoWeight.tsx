import React from 'react';
import { Fab } from '@material-ui/core';
import { ArrowUpward, ArrowDownward } from '@material-ui/icons';
import { writeBase64ToCanvas } from '../utils';
import TensorCanvas from './TensorCanvas';

interface PopupContentNoWeightProps {
  input: string[];
  output: string[];
  isOpen: boolean;
  info: LayerInfo;
}

interface PopupContentNoWeightState {
  idx: number;
}

class PopupContentNoWeight extends React.Component<
  PopupContentNoWeightProps,
  PopupContentNoWeightState
> {
  state: PopupContentNoWeightState = {
    idx: 0,
  };

  inputRef: React.RefObject<HTMLCanvasElement> = React.createRef();
  outputRef: React.RefObject<HTMLCanvasElement> = React.createRef();

  componentDidMount() {
    setTimeout(() => this.rewriteCanvas(null, null), 100);
  }

  componentDidUpdate(
    prevProps: PopupContentNoWeightProps,
    prevState: PopupContentNoWeightState
  ) {
    if (!prevProps.isOpen && this.props.isOpen) {
      setTimeout(() => this.rewriteCanvas(prevProps, prevState), 100);
    } else {
      this.rewriteCanvas(null, null);
    }
  }

  writeToCanvas = (
    tensor: 'input' | 'output' | 'weight',
    size: number,
    data: string
  ) => {
    const ref = (tensor + 'Ref') as 'inputRef' | 'outputRef';
    const canvas = this[ref].current!;
    writeBase64ToCanvas(canvas, data, size);
  };

  rewriteCanvas = (
    prevProps: PopupContentNoWeightProps | null,
    prevState: PopupContentNoWeightState | null
  ) => {
    if (
      this.props.input !== undefined &&
      this.props.input.length > 0 &&
      this.props.output !== undefined &&
      this.props.output.length > 0
    ) {
      const justOpen = prevProps && !prevProps.isOpen && this.props.isOpen;
      const toUpdateInputCanvas =
        !prevState || prevState.idx !== this.state.idx || justOpen;

      if (toUpdateInputCanvas) {
        const inputData = this.props.input[this.state.idx];
        this.writeToCanvas('input', 300, inputData);
        const outputData = this.props.output[this.state.idx];
        this.writeToCanvas('output', 300, outputData);
      }
    }
  };

  changeIndexFactory = (direction: 1 | -1) => {
    return () => {
      this.setState({ idx: this.state.idx + direction });
    };
  };

  render() {
    return (
      <div style={{ display: 'flex', flexDirection: 'row' }}>
        <TensorCanvas reference={this.inputRef} size={300} margin={30} />
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <Fab
            disabled={this.state.idx <= 0}
            onClick={this.changeIndexFactory(-1)}
            style={{ margin: '20px' }}
          >
            <ArrowUpward />
          </Fab>
          {`Channel ${this.state.idx}/${this.props.input.length}`}
          <Fab
            disabled={this.state.idx >= this.props.input.length - 1}
            onClick={this.changeIndexFactory(1)}
            style={{ margin: '20px' }}
          >
            <ArrowDownward />
          </Fab>
        </div>
        <TensorCanvas reference={this.outputRef} size={300} margin={30} />
      </div>
    );
  }
}

export default PopupContentNoWeight;
