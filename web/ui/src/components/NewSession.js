import * as React from 'react';
import Card from '@mui/material/Card';
import CardActions from '@mui/material/CardActions';
import CardContent from '@mui/material/CardContent';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import {
  Box,
  Checkbox,
  FormControlLabel,
  FormGroup,
  TextField,
} from '@mui/material';

export default function NewSession() {
  return (
    <Card sx={{ mt: 2 }}>
      <CardContent>
        <Typography gutterBottom variant="h5" component="div">
          Load or Create a New Dataset
        </Typography>
        <FormGroup>
          <FormControlLabel
            control={<Checkbox defaultChecked />}
            label="Include <start> / <end> tokens"
          />
          <FormControlLabel
            control={<Checkbox defaultChecked />}
            label="Include punctuation"
          />
          <FormControlLabel
            control={<Checkbox defaultChecked />}
            label="Allow All"
          />
          <FormControlLabel
            control={<Checkbox defaultChecked />}
            label="Use Exact Word"
          />
          <FormControlLabel
            control={<Checkbox defaultChecked />}
            label="Use Lemma"
          />
          <FormControlLabel
            control={<Checkbox defaultChecked />}
            label="Add Word to Vocabulary"
          />
          <FormControlLabel
            control={<Checkbox defaultChecked />}
            label="Add Lemma to Vocabulary"
          />
        </FormGroup>
      </CardContent>
      <CardActions>
        <Button size="small">Create</Button>
      </CardActions>
    </Card>
  );
}
