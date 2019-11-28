import React from 'react';
import { withStyles, Theme } from '@material-ui/core/styles';
import Dialog from '@material-ui/core/Dialog';
import MuiDialogTitle from '@material-ui/core/DialogTitle';
import IconButton from '@material-ui/core/IconButton';
import CloseIcon from '@material-ui/icons/Close';
import Typography from '@material-ui/core/Typography';
import { Styles } from '@material-ui/styles/withStyles';
import { Fab } from '@material-ui/core';
import { ArrowUpward, ArrowDownward } from '@material-ui/icons';

const styles: Styles<Theme, {}, 'root' | 'closeButton'> = (theme: Theme) => ({
  root: {
    margin: 0,
    padding: theme.spacing(2),
  },
  closeButton: {
    position: 'absolute',
    right: theme.spacing(1),
    top: theme.spacing(1),
    color: theme.palette.grey[500],
  },
});

const DialogTitle = withStyles(styles)((props: any) => {
  const { children, classes, onClose, ...other } = props;
  return (
    <MuiDialogTitle disableTypography className={classes.root} {...other}>
      <Typography variant="h6">{children}</Typography>
      <IconButton
        aria-label="close"
        className={classes.closeButton}
        onClick={onClose}
      >
        <CloseIcon />
      </IconButton>
    </MuiDialogTitle>
  );
});

interface LayerTracePopupProps {
  input: any[];
  output: any[];
  weights: any[][];
  isOpen: boolean;
  onClose: () => void;
}

interface LayerTracePopupState {
  inputIdx: number;
  outputIdx: number;
}

class LayerTracePopup extends React.Component<
  LayerTracePopupProps,
  LayerTracePopupState
> {
  state: LayerTracePopupState = {
    inputIdx: 1,
    outputIdx: 1,
  };

  inputRef: React.RefObject<HTMLCanvasElement> = React.createRef();
  outputRef: React.RefObject<HTMLCanvasElement> = React.createRef();
  weightRef: React.RefObject<HTMLCanvasElement> = React.createRef();

  componentDidMount() {
    // this.rewriteCanvas(null);
    // this.setState({
    //   inputIdx: 0,
    //   outputIdx: 0,
    // });
    setTimeout(() => this.rewriteCanvas(null, null), 100);
  }

  componentDidUpdate(
    prevProps: LayerTracePopupProps,
    prevState: LayerTracePopupState
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
    console.log(this[ref]);
    const canvas = this[ref].current!;
    const ctx = canvas.getContext('2d')!;
    ctx.imageSmoothingEnabled = false;
    const image = new Image();
    image.onload = () => ctx.drawImage(image, 0, 0, size, size);
    image.src = 'data:image/jpeg;base64,' + data;
  };

  rewriteCanvas = (
    prevProps: LayerTracePopupProps | null,
    prevState: LayerTracePopupState | null
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
        >
          <ArrowUpward />
        </Fab>
        <canvas ref={this[ref]} width="300" height="300" />
        {this.state[field]}
        <Fab
          disabled={this.state[field] >= this.props[tensor].length - 1}
          onClick={this.changeIndexFactory(tensor, 1)}
        >
          <ArrowDownward />
        </Fab>
      </div>
    );
  };

  render() {
    return (
      <Dialog
        onClose={this.props.onClose}
        aria-labelledby="customized-dialog-title"
        open={this.props.isOpen}
      >
        <DialogTitle id="customized-dialog-title" onClose={this.props.onClose}>
          Modal title
        </DialogTitle>
        <div style={{ display: 'flex', flexDirection: 'row' }}>
          {this.makeSelector('input')}
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <canvas ref={this.weightRef} width="200" height="200" />
          </div>
          {this.makeSelector('output')}
        </div>
      </Dialog>
    );
  }
}

export default LayerTracePopup;
