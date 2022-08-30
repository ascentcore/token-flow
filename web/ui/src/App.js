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
  Slider,
  TextField,
} from '@mui/material';

import Graph from './components/Graph';

const mdTheme = createTheme();

function DashboardContent() {
  const [initial, setIntial] = React.useState(false);
  const [contexts, setContexts] = React.useState(null);
  const [textValue, setTextValue] = React.useState('');
  const [currentState, setCurrentState] = React.useState('');
  const [stimulate, setStimulate] = React.useState(true);
  const [threshold, setThreshold] = React.useState(0.2);

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
        </Box>
        {/* {initial && <NewSession />} */}
        <Grid container spacing={2} sx={{ mt: 2 }}>
          {contexts &&
            contexts.map((context) => (
              <Grid item xs key={context}>
                <Graph
                  context={context}
                  threshold={threshold}
                  state={currentState}
                  onNodeClick={onNodeClick}
                />
              </Grid>
            ))}
        </Grid>
      </Box>
    </ThemeProvider>
  );
}

export default function Dashboard() {
  return <DashboardContent />;
}
