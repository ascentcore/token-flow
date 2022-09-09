import * as React from 'react';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import * as d3 from 'd3';

import axios from 'axios';
import {
  Button,
  Checkbox,
  FormControlLabel,
  Grid,
  Modal,
  Slider,
  Switch,
  TextField,
  Typography,
} from '@mui/material';

import Graph from './components/Graph';
import RadialGraph from './components/RadialGraph';

const mdTheme = createTheme();

const style = {
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: 600,
  backgroundColor: '#FFF',
  boxShadow: 24,
  pt: 2,
  px: 4,
  pb: 3,
};

function DashboardContent() {
  const [initial, setIntial] = React.useState(false);
  const [contexts, setContexts] = React.useState(null);
  const [textValue, setTextValue] = React.useState('');
  const [currentState, setCurrentState] = React.useState('');
  const [stimulate, setStimulate] = React.useState(true);
  const [threshold, setThreshold] = React.useState(0.2);
  const [selectedContext, setSelectedContext] = React.useState(null);
  const [knowledge, setKnowledge] = React.useState(null);

  const [radial, setRadial] = React.useState(true);

  React.useEffect(() => {
    axios
      .get('http://localhost:8081/contexts')
      .then((response) => {
        const { data } = response;
        setContexts(data);
      })
      .catch((err) => {
        console.log(err);
      });
  }, []);

  function handleChange(e) {
    setTextValue(e.target.value);
  }

  function keyPress(e) {
    if (e.keyCode === 13) {
      const sendValue = e.target.value.trim();
      if (sendValue !== '') {
        setTextValue('');
        axios
          .post(
            `http://localhost:8081/${stimulate ? 'stimulate' : 'add_text'}`,
            { text: sendValue }
          )
          .then((response) => {
            const { data } = response;
            setCurrentState(data);
          });
      }
    }
  }

  function addKnolwedgeToContext(context, text) {
    setSelectedContext(null);
    axios
      .post(`http://localhost:8081/add_text/${context}`, {
        text,
      })
      .then((response) => {
        setKnowledge(null);
        const { data } = response;
        setCurrentState(data);
      });
  }

  function resetStimuli() {
    axios.post(`http://localhost:8081/reset_stimuli`).then((response) => {
      const { data } = response;
      setCurrentState(data);
    });
  }

  function onNodeClick(id) {
    axios
      .post('http://localhost:8081/stimulate', { text: id })
      .then((response) => {
        const { data } = response;
        setCurrentState(data);
      });
  }

  return (
    <ThemeProvider theme={mdTheme}>
      <Box sx={{ display: 'flex', flexDirection: 'column', m: 4 }}>
        <CssBaseline />
        <Box sx={{ display: 'flex', flexDirection: 'row' }}>
          <TextField
            fullWidth
            label="Input Text"
            disabled={initial}
            value={textValue}
            onKeyDown={keyPress}
            onChange={handleChange}
          />
          <FormControlLabel
            control={
              <Checkbox
                checked={stimulate}
                onChange={(e) => setStimulate(e.target.checked)}
              />
            }
            label="Stimulate"
          />
          <Button variant="contained" onClick={() => resetStimuli()}>
            Reset Stimuli
          </Button>
        </Box>
        <Box sx={{ display: 'flex', flexDirection: 'column' }}>
          <Slider
            value={threshold}
            min={0}
            max={1}
            valueLabelDisplay="on"
            step={0.01}
            onChange={(e) => setThreshold(e.target.value)}
          ></Slider>
          <Switch checked={radial} onChange={(e) => setRadial(e.target.checked)} />
        </Box>
        {/* {initial && <NewSession />} */}
        <Grid container spacing={2} sx={{ mt: 2 }}>
          {contexts &&
            contexts.map((context) => (
              <Grid item xs key={context}>
                {radial && (
                  <RadialGraph
                    context={context}
                    threshold={threshold}
                    state={currentState}
                    onNodeClick={onNodeClick}
                    selectContext={setSelectedContext}
                  />
                )}
                {!radial && (
                  <Graph
                    context={context}
                    threshold={threshold}
                    state={currentState}
                    onNodeClick={onNodeClick}
                    selectContext={setSelectedContext}
                  />
                )}
              </Grid>
            ))}
        </Grid>
      </Box>
      <Modal open={selectedContext !== null}>
        <div style={style}>
          <Box sx={{ display: 'flex', flexDirection: 'column', m: 2 }}>
            <Typography sx={{ mb: 2 }}>
              Add information to {selectedContext}
            </Typography>
            <TextField
              label="Knowledge"
              multiline
              rows={8}
              sx={{ mb: 2 }}
              value={knowledge}
              onChange={(e) => setKnowledge(e.target.value)}
            />
            <Box sx={{ mt: 2, display: 'flex', flexDirection: 'row-reverse' }}>
              <Button
                variant="contained"
                color="primary"
                onClick={() =>
                  addKnolwedgeToContext(selectedContext, knowledge)
                }
              >
                Add
              </Button>
              <Button
                variant="link"
                onClick={() => {
                  setKnowledge(null);
                  setSelectedContext(null);
                }}
              >
                Cancel
              </Button>
            </Box>
          </Box>
        </div>
      </Modal>
    </ThemeProvider>
  );
}

export default function Dashboard() {
  return <DashboardContent />;
}