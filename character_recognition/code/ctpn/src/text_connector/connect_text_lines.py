'''
Date: 2021-08-15 16:11:14
LastEditors: xgy
LastEditTime: 2021-08-15 16:12:39
FilePath: \code\ctpn\src\text_connector\connect_text_lines.py
'''

import numpy as np
from src.text_connector.utils import clip_boxes, fit_y
from src.text_connector.get_successions import get_successions

def connect_text_lines(text_proposals, scores, size):
    """
    Connect text lines

    Args:
        text_proposals(numpy.array): Predict text proposals.
        scores(numpy.array): Bbox predicts scores.
        size(numpy.array): Image size.
    Returns:
        text_recs(numpy.array): Text boxes after connect.
    """
    graph = get_successions(text_proposals, scores, size)
    text_lines = np.zeros((len(graph), 5), np.float32)
    for index, indices in enumerate(graph):
        text_line_boxes = text_proposals[list(indices)]
        x0 = np.min(text_line_boxes[:, 0])
        x1 = np.max(text_line_boxes[:, 2])

        offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5

        lt_y, rt_y = fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
        lb_y, rb_y = fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)

        # the score of a text line is the average score of the scores
        # of all text proposals contained in the text line
        score = scores[list(indices)].sum() / float(len(indices))

        text_lines[index, 0] = x0
        text_lines[index, 1] = min(lt_y, rt_y)
        text_lines[index, 2] = x1
        text_lines[index, 3] = max(lb_y, rb_y)
        text_lines[index, 4] = score

    text_lines = clip_boxes(text_lines, size)

    text_recs = np.zeros((len(text_lines), 9), np.float)
    index = 0
    for line in text_lines:
        xmin, ymin, xmax, ymax = line[0], line[1], line[2], line[3]
        text_recs[index, 0] = xmin
        text_recs[index, 1] = ymin
        text_recs[index, 2] = xmax
        text_recs[index, 3] = ymax
        text_recs[index, 4] = line[4]
        index = index + 1
    return text_recs
