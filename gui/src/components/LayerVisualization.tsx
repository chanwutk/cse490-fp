import React from 'react';
import {
  makeRequest,
  drawArrow,
  layerRootStyle,
  layerThinDivStyle,
} from '../utils';
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
import CircularProgress from '@material-ui/core/CircularProgress';
import CloseIcon from '@material-ui/icons/Close';

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
  imageData?: string;
}

interface LayerVisualizationState {
  isReady: boolean;
  isPopupOpen: boolean;
  input?: string[];
  weights?: string[][];
  output?: string[];
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

  componentDidUpdate(prevProps: LayerVisualizationProps) {
    if (prevProps.imageData !== this.props.imageData) {
      this.setState({ isReady: false });
    }
  }

  componentDidMount() {
    drawArrow(this.canvas.current!);
  }

  onClick = async () => {
    if (!this.state.isReady) {
      await this.setState({ isPopupOpen: true });
      const result = await makeRequest(`trace/${this.props.idx}`, JSON.parse);
      this.setState({
        isReady: true,
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

  commonPopupProps = () => {
    return {
      input: this.state.input!,
      output: this.state.output!,
      isOpen: this.state.isPopupOpen,
      info: this.props.info,
    };
  };

  render() {
    return (
      <div style={layerRootStyle}>
        <div style={{ justifyContent: 'right', ...layerThinDivStyle }}>
          <div style={{ marginRight: '10px' }}>{this.props.info.str}</div>
        </div>
        <canvas ref={this.canvas} width="80" height="80" />
        <div style={layerThinDivStyle}>
          <Fab
            variant="extended"
            color={this.props.info.type === 'Conv2d' ? 'secondary' : 'primary'}
            style={{ left: '10px' }}
            onClick={this.onClick}
          >
            Trace this Layer!
          </Fab>
        </div>
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
            {this.state.isReady ? (
              this.props.info.type === 'Conv2d' ? (
                <PopupContentWithWeight
                  weights={this.state.weights!}
                  {...this.commonPopupProps()}
                />
              ) : (
                <PopupContentNoWeight {...this.commonPopupProps()} />
              )
            ) : (
              <div>
                <CircularProgress
                  style={{
                    margin: '20px 350px 20px 350px',
                  }}
                />
              </div>
            )}
          </div>
        </Dialog>
      </div>
    );
  }
}

export default LayerVisualization;
