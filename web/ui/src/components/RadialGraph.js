import {
  Button,
  Card,
  CardContent,
  Chip,
  Typography,
} from '@mui/material';
import * as React from 'react';
import axios from 'axios';
import * as d3 from 'd3';
import { Box } from '@mui/system';

export default function Graph(props) {
  const { context, state, onNodeClick, threshold, selectContext } = props;
  const containerRef = React.useRef(null);
  const [data, setData] = React.useState(null);
  const [svgContainer, setSvgContainer] = React.useState(null);
  const [sim, setSim] = React.useState(null);
  const [nodes, setNodes] = React.useState(null);
  const [links, setLinks] = React.useState(null);
  const [topTokens, setTopTokens] = React.useState([]);

  function processTopTokens(nodes) {
    nodes.sort((a, b) => b.s - a.s);

    setTopTokens(nodes.slice(0, 10));
  }

  function onClick() {
    const elems = d3.select(this).data();
    if (elems && elems.length > 0) {
      const { id } = elems[0];
      onNodeClick(id);
    }
  }

  function doNode(node) {
    node.on('click', onClick);

    node.attr('class', 'node');
    node.attr('opacity', (d) => {
      return d.s >= threshold ? 1 : 0;
    });
    node
      .append('circle')
      .attr('class', 'stimulus')
      .attr('fill', 'rgba(255,0,0,0.2)')
      .attr('r', (d) => (Math.pow(2, d.s) - 1) * 30);
    node.append('circle').attr('r', 2);
    node
      .append('text')
      .text((d) => d.id)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('alignment-baseline', 'baseline')
      .attr('transform', 'translate(0, -6)');
  }

  function doLine(line) {
    line.attr('stroke', 'rgba(0,0,0,.2)').attr('stroke-width', (d) => d.weight);
  }

  React.useEffect(() => {
    axios
      .get(`http://localhost:8081/context/${context}/graph`)
      .then((response) => {
        const { data: graph } = response;
        processTopTokens(graph.nodes);
        setData(graph);

        const rootContainer = d3.select(containerRef.current);
        rootContainer.selectAll('*').remove();
        const svg = rootContainer.append('g');
        const width = containerRef.current.getBoundingClientRect().width;
        const height = +rootContainer.attr('height');

        setSvgContainer(svg);

        rootContainer.call(
          d3.zoom().on('zoom', function (event) {
            svg.attr('transform', event.transform);
          })
        );

        const simulation = d3
          .forceSimulation()
          .force(
            'radial',
            d3.forceRadial(function (d) {
              return (2 - Math.pow(2, d.s)) * (width / 2);
            })
            // .strength(0.1)
          )
          .force(
            'charge',
            d3.forceManyBody().strength((d) => -d.s * 60)
          )
          .force(
            'link',
            d3
              .forceLink()
              .id((d) => d.id)
              // .distance(30)
              .strength(0.001)
          );
        // // .force('x', d3.forceX())
        // // .force('y', d3.forceY())
        // .force('center', d3.forceCenter(width / 2, height / 2));

        var link = svg
          .append('g')
          .attr('class', 'links')
          .selectAll('line')
          .data(graph.links)
          .enter()
          .append('line');

        doLine(link);

        var node = svg
          .append('g')
          .attr('class', 'nodes')
          .selectAll('g')
          .data(graph.nodes)
          .enter()
          .append('g');

        doNode(node);

        setNodes(node);
        setLinks(link);

        const ticked = () => {
          (svgContainer || svg)
            .selectAll('line')

            .attr('x1', function (d) {
              return d.source.x;
            })
            .attr('y1', function (d) {
              return d.source.y;
            })
            .attr('x2', function (d) {
              return d.target.x;
            })
            .attr('y2', function (d) {
              return d.target.y;
            });

          //

          (svgContainer || svg).selectAll('.node').attr('transform', (d) => {
            return `translate(${d.x}, ${d.y})`;
          });
        };

        simulation.nodes(graph.nodes).on('tick', ticked);
        simulation.force('link').links(graph.links);

        setSim(simulation);
      })
      .catch((err) => {
        console.log(err);
      });
    return () => {
      adjustOpacity();
    };
  }, [context]);

  function adjustOpacity() {
    if (svgContainer) {
      svgContainer
        .selectAll('.node')     
        .transition()
        .duration(300)
        .attr('opacity', (d) => {
          return d.s >= threshold ? 1 : 0;
        });

      svgContainer
        .selectAll('.stimulus')
        .transition()
        .duration(300)
        .attr('r', (d) => (Math.pow(2, d.s) - 1) * 30);

      svgContainer
        .selectAll('line')
        .transition()
        .duration(300)
        .attr('stroke-opacity', (d) =>
          Math.max(d.source.s, d.target.s) - threshold > 0
            ? Math.max(d.source.s, d.target.s)
            : 0
        );
    }
  }

  React.useEffect(() => {
    adjustOpacity();
  }, [threshold]);

  React.useEffect(() => {
    axios
      .get(`http://localhost:8081/context/${context}/graph`)
      .then((response) => {
        const { data: graph } = response;
        processTopTokens(graph.nodes);
        if (nodes) {
          const old = new Map(nodes.data().map((d) => [d.id, d]));
          const updatedNodes = graph.nodes.map((d) => {
            const result = Object.assign(old.get(d.id) || {}, d);
            result.s = d.s;
            return result;
          });
          const updatedLinks = graph.links.map((d) => Object.assign({}, d));
          sim.nodes(updatedNodes);
          sim.force('link').links(updatedLinks);
          sim.alpha(1).restart();

          const newNode = nodes
            .data(updatedNodes, (d) => d.id)
            .join((enter) => {
              enter = enter.append('g');

              enter.on('click', onClick);

              enter.attr('class', 'node');
              doNode(enter);

              return enter;
            });

          const newLinks = links
            .data(graph.links, (d) => {
              d.source = updatedNodes.find((n) => n.id === d.source);
              d.target = updatedNodes.find((n) => n.id === d.target);
            })
            .join('line');

          doLine(newLinks);
          setNodes(newNode);
          setLinks(newLinks);
          sim.alpha(1).restart();
          adjustOpacity();
        }
      });
  }, [state]);

  return (
    <Card>
      <CardContent>
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'row',
            justifyContent: 'space-between',
          }}
        >
          <Typography gutterBottom variant="h5" component="div">
            {context}
          </Typography>
          <Button
            size="small"
            variant="outline"
            onClick={() => selectContext(context)}
          >
            Add Knowledge
          </Button>
        </Box>

        <svg id="content" width="100%" height="500" ref={containerRef}></svg>
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
