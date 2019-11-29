import React from 'react';
import { Fab } from '@material-ui/core';
import { ArrowUpward, ArrowDownward } from '@material-ui/icons';
import { writeBase64ToCanvas } from '../utils';

interface PopupContentWithWeightProps {
  input: any[];
  output: any[];
  weights: any[][];
  isOpen: boolean;
  info: LayerInfo;
}

interface PopupContentWithWeightState {
  inputIdx: number;
  outputIdx: number;
}

class PopupContentWithWeight extends React.Component<
  PopupContentWithWeightProps,
  PopupContentWithWeightState
> {
  state: PopupContentWithWeightState = {
    inputIdx: 0,
    outputIdx: 0,
  };

  inputRef: React.RefObject<HTMLCanvasElement> = React.createRef();
  outputRef: React.RefObject<HTMLCanvasElement> = React.createRef();
  weightRef: React.RefObject<HTMLCanvasElement> = React.createRef();

  componentDidMount() {
    setTimeout(() => this.rewriteCanvas(null, null), 100);
  }

  componentDidUpdate(
    prevProps: PopupContentWithWeightProps,
    prevState: PopupContentWithWeightState
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
    const ref = (tensor + 'Ref') as 'inputRef' | 'outputRef' | 'weightRef';
    const canvas = this[ref].current!;
    writeBase64ToCanvas(canvas, data, size);
  };

  rewriteCanvas = (
    prevProps: PopupContentWithWeightProps | null,
    prevState: PopupContentWithWeightState | null
  ) => {
    if (
      this.props.input !== undefined &&
      this.props.input.length > 0 &&
      this.props.output !== undefined &&
      this.props.output.length > 0
    ) {
      const justOpen = prevProps && !prevProps.isOpen && this.props.isOpen;
      const toUpdateInputCanvas =
        !prevState || prevState.inputIdx !== this.state.inputIdx || justOpen;
      const toUpdateOutputCanvas =
        !prevState || prevState.outputIdx !== this.state.outputIdx || justOpen;

      if (toUpdateInputCanvas) {
        const data = this.props.input[this.state.inputIdx];
        this.writeToCanvas('input', 300, data);
      }

      if (toUpdateOutputCanvas) {
        const data = this.props.output[this.state.outputIdx];
        this.writeToCanvas('output', 300, data);
      }

      if (toUpdateInputCanvas || toUpdateOutputCanvas) {
        const data = this.props.weights[this.state.outputIdx][
          this.state.inputIdx
        ];
        this.writeToCanvas('weight', 200, data);
      }
    }
  };

  changeIndexFactory = (tensor: 'input' | 'output', direction: 1 | -1) => {
    const field = (tensor + 'Idx') as 'inputIdx' | 'outputIdx';
    return () => {
      this.setState({ [field]: this.state[field] + direction } as any);
    };
  };

  makeSelector = (tensor: 'input' | 'output') => {
    const field = (tensor + 'Idx') as 'inputIdx' | 'outputIdx';
    const ref = (tensor + 'Ref') as 'inputRef' | 'outputRef';

    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
        }}
      >
        <Fab
          disabled={this.state[field] <= 0}
          onClick={this.changeIndexFactory(tensor, -1)}
          style={{ marginTop: '20px' }}
        >
          <ArrowUpward />
        </Fab>
        <canvas
          ref={this[ref]}
          width="300"
          height="300"
          style={{ margin: '30px' }}
        />
        <Fab
          disabled={this.state[field] >= this.props[tensor].length - 1}
          onClick={this.changeIndexFactory(tensor, 1)}
          style={{ marginBottom: '20px' }}
        >
          <ArrowDownward />
        </Fab>
      </div>
    );
  };

  render() {
    return (
      <div style={{ display: 'flex', flexDirection: 'row' }}>
        {this.makeSelector('input')}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
          }}
        >
          {`Kernel[${this.state.inputIdx}][${this.state.outputIdx}]`}
          <canvas
            ref={this.weightRef}
            width="200"
            height="200"
            style={{ margin: '10px' }}
          />
        </div>
        {this.makeSelector('output')}
      </div>
    );
  }
}

export default PopupContentWithWeight;
