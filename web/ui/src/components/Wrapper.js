import {
    Badge,
    Button,
    Card,
    CardContent,
    Checkbox,
    Chip,
    FormControlLabel,
    Paper,
    Typography,
  } from '@mui/material';
  import * as React from 'react';
  import axios from 'axios';
  import * as d3 from 'd3';
  import { Box } from '@mui/system';
  
  export default function Wrapper(props) {
    
  
    return (
      <Card>
        <CardContent>
          {topTokens.map((token) => (
            <Chip
              key={token.id}
              sx={{ mr: 1, mb: 1 }}
              onClick={() => onNodeClick(token.id)}
              label={`${token.id}: ${Math.round(token.s * 100) / 100}`}
              size="small"
              color={'primary'}
            />
          ))}
        </CardContent>
      </Card>
    );
  }
  