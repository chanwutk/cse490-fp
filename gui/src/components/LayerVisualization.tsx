import React from 'react';
import { makeRequest } from '../utils';
import {
  Fab,
  Dialog,
  Theme,
  withStyles,
  Typography,
  IconButton,
} from '@material-ui/core';
import MuiDialogTitle from '@material-ui/core/DialogTitle';
import PopupContentWithWeight from './PopupContentWithWeight';
import PopupContentNoWeight from './PopupContentNoWeight';
import { Styles } from '@material-ui/styles/withStyles';
import CloseIcon from '@material-ui/icons/Close';

const rootStyle: React.CSSProperties = {
  padding: '5px',
  display: 'flex',
  alignItems: 'center',
  flexDirection: 'row',
};

const thinDivStyle: React.CSSProperties = {
  width: '100%',
  display: 'flex',
  alignItems: 'center',
};

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

interface LayerVisualizationProps {
  info: LayerInfo;
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
          <div style={{ marginRight: '10px' }}>{this.props.info.str}</div>
        </div>
        <canvas ref={this.canvas} width="80" height="80" />
        <div style={thinDivStyle}>
          <Fab
            variant="extended"
            color="primary"
            style={{
              left: '10px',
            }}
            onClick={this.onClick}
          >
            Trace this Layer!
          </Fab>
        </div>
        {this.state.isReady ? (
          <Dialog
            onClose={this.closePopup}
            aria-labelledby="customized-dialog-title"
            open={this.state.isPopupOpen}
            maxWidth="lg"
          >
            <DialogTitle id="customized-dialog-title" onClose={this.closePopup}>
              {this.props.info.str}
            </DialogTitle>
            <div style={{ display: 'flex', flexDirection: 'row' }}>
              {this.props.info.type === 'Conv2d' ? (
                <PopupContentWithWeight
                  input={this.state.input}
                  output={this.state.output}
                  weights={this.state.weights}
                  isOpen={this.state.isPopupOpen}
                  info={this.props.info}
                />
              ) : (
                <PopupContentNoWeight
                  input={this.state.input}
                  output={this.state.output}
                  isOpen={this.state.isPopupOpen}
                  info={this.props.info}
                />
              )}
            </div>
          </Dialog>
        ) : null}
      </div>
    );
  }
}

export default LayerVisualization;
