from Rect import Rect
from xml.dom import minidom
from xml.etree import ElementTree as ET


class GTElement(Rect):
    """
    An abstract class that represents a Ground Truth Bounding Box
    """

    def __init__(self, x0=0, y0=0, x1=1, y1=1):
        super().__init__(x0, y0, x1, y1)


class Row(GTElement):
    """
    A class that represents a Row seperator in table (rectangle with height=1)
    """

    def __init__(self, x0=0, y0=0, x1=1):
        super().__init__(x0, y0, x1, y0 + 1)


class Column(GTElement):
    """
    A class that represents a Column seperator in table (rectangle with width=1)
    """

    def __init__(self, x0=0, y0=0, y1=1):
        super().__init__(x0, y0, x0 + 1, y1)


class Cell(GTElement):
    """
    A class that represents a Cell in table
    """

    def __init__(self, x0=0, y0=0, x1=1, y1=1, startRow=-1, startCol=-1):
        super().__init__(x0, y0, x1, y1)

        self.startRow = startRow
        self.startCol = startCol

        self.endRow = startRow
        self.endCol = startCol

        self.words = []

        self.dontCare = False

    # def assumeDontCare(self):
    #     if self.endRow - self.startRow < 0 or self.endCol - self.startCol < 0:
    #         self.dontCare = True
    #     else:
    #         self.dontCare = False

    def getCenter(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


HORIZONTAL = 0
VERTICAL = 1
UNKNOWN = 2


class Table(GTElement):
    """
    A class for representing a Ground Truth table.
    """

    def __init__(self, x0=0, y0=0, x1=1, y1=1):
        super().__init__(x0, y0, x1, y1)
        self.orientation = UNKNOWN
        self.gtSpans = []
        self.cells = []
        self.gtRows = []
        self.gtCols = []
        self.gtCells = None

    def __str__(self):
        return "Width: " + str(self.x2) + ", Height: " + str(self.y2)

    def __repr__(self):
        print("======================x======================")
        _sum_items = sum([len(item) for item in self.gtCells])
        print("Table Contains " + str(_sum_items) + " Cells")
        print("Cell Details are as following: ")
        print("----------------------x----------------------")
        for item in self.gtCells:
            print(item)

    def remove(self, elem):
        if elem in self.gtRows:
            self.gtRows.remove(elem)
        elif elem in self.gtCols:
            self.gtCols.remove(elem)
        elif elem in self.gtSpans:
            self.gtSpans.remove(elem)
        self.evaluateCells()

    def getCellAtPoint(self, p):
        if len(self.gtCells) == 0:
            return None
        else:
            for i in range(len(self.gtCells)):
                for j in range(len(self.gtCells[i])):
                    cell = self.gtCells[i][j]
                    if (
                        p[0] >= cell.x1
                        and p[0] <= cell.x2
                        and p[1] >= cell.y1
                        and p[1] <= cell.y2
                    ):
                        return cell
        return None

    def addSpan(self, elem):
        self.gtSpans.append(elem)
        self.evaluateCells()

    def removeSpan(self, elem):
        self.gtSpans.remove(elem)
        self.evaluateCells()

    def evaluateCells(self):
        self.evaluateInitialCells()
        returnVal = [False for i in range(len(self.gtSpans))]
        for i, elem in enumerate(self.gtSpans):
            if isinstance(elem, Column):
                returnVal[i] = self.addColSpan((elem.x1, elem.y1), (elem.x2, elem.y2))
            elif isinstance(elem, Row):
                returnVal[i] = self.addRowSpan((elem.x1, elem.y1), (elem.x2, elem.y2))

        for i in range(len(self.gtSpans) - 1, -1, -1):
            if returnVal[i] is False:
                self.gtSpans.remove(self.gtSpans[i])

    def addRowSpan(self, p1, p2):
        startCell = self.getCellAtPoint(p1)
        endCell = self.getCellAtPoint(p2)
        if (
            startCell is None
            or endCell is None
            or startCell.startRow != endCell.startRow
            or startCell.endRow != endCell.endRow
            or startCell == endCell
        ):
            # print("Cant add Row Span: for " + str(p1) + " ,and " + str(p2))
            return False

        startCell.endCol = endCell.endCol
        for i in range(startCell.startCol + 1, endCell.endCol + 1):
            temp = self.gtCells[startCell.startRow][i]
            temp.dontCare = True
            startCell.set_x2(temp.x2)
            if temp.y2 > startCell.y2:
                startCell.set_y2(temp.y2)

            for j in range(startCell.startRow + 1, startCell.endRow + 1):
                self.gtCells[j][i].dontCare = True
        return True

    def addColSpan(self, p1, p2):
        startCell = self.getCellAtPoint(p1)
        endCell = self.getCellAtPoint(p2)
        if (
            startCell is None
            or endCell is None
            or startCell.startCol != endCell.startCol
            or startCell.endCol != endCell.endCol
            or startCell == endCell
        ):
            # print("Cant add Col Span: for " + str(p1) + " ,and " + str(p2))
            return False

        startCell.endRow = endCell.endRow
        for i in range(startCell.startRow + 1, endCell.endRow + 1):
            temp = self.gtCells[i][startCell.startCol]
            temp.dontCare = True
            startCell.set_y2(temp.y2)
            if temp.x2 > startCell.x2:
                startCell.set_x2(temp.x2)

            for j in range(startCell.startCol + 1, startCell.endCol + 1):
                self.gtCells[i][j].dontCare = True
        return True

    def populateSpansFromCells(self):
        numRows = len(self.gtRows) + 1
        numCols = len(self.gtCols) + 1
        self.gtCells = [[None for j in range(numCols)] for i in range(numRows)]
        self.gtSpans.clear()

        self.cells.sort(key=lambda cell: (cell.y1, cell.x1))

        if len(self.cells) != numRows * numCols:
            print("Arrays don't match for cells")
            return

        for cell in self.cells:
            self.gtCells[cell.startRow][cell.startCol] = cell

        if len(self.gtRows) == 0:
            rowCenters = [(self.y1 + self.y2) / 2]
        else:
            rowCenters = [
                (self.gtRows[i].y1 + self.gtRows[i + 1].y1) // 2
                for i in range(len(self.gtRows) - 1)
            ]
            rowCenters = (
                [(self.gtRows[0].y1 + self.y1) // 2]
                + rowCenters
                + [(self.gtRows[-1].y1 + self.y2) // 2]
            )

        if len(self.gtCols) == 0:
            colCenters = [(self.x1 + self.x2) / 2]
        else:
            colCenters = [
                (self.gtCols[i].x1 + self.gtCols[i + 1].x1) // 2
                for i in range(len(self.gtCols) - 1)
            ]
            colCenters = (
                [(self.gtCols[0].x1 + self.x1) // 2]
                + colCenters
                + [(self.gtCols[-1].x1 + self.x2) // 2]
            )

        for i in range(numRows):
            for j in range(numCols):
                cell = self.gtCells[i][j]

                if cell.dontCare is False:
                    if cell.startCol != cell.endCol:
                        cell.set_x2(self.gtCells[i][j + 1].x1)
                        for i1 in range(cell.startRow, cell.endRow + 1):
                            y1 = rowCenters[i1]
                            # print(cell.startCol, cell.endCol)
                            x1 = colCenters[cell.startCol]
                            x2 = colCenters[cell.endCol]
                            # print(x1, y1, x2)
                            self.gtSpans.append(Row(x1, y1, x2))
                    if cell.startRow != cell.endRow:
                        x1 = colCenters[cell.startCol]
                        y1 = rowCenters[cell.startRow]
                        y2 = rowCenters[cell.endRow]
                        self.gtSpans.append(Column(x1, y1, y2))

    def evaluateInitialCells(self):
        self.gtRows.sort(key=lambda x: x.y1)
        self.gtCols.sort(key=lambda x: x.x1)

        numRows = len(self.gtRows) + 1
        numCols = len(self.gtCols) + 1

        self.gtCells = [[None for j in range(numCols)] for i in range(numRows)]
        self.cells = []

        l, t, r, b = 0, 0, 0, 0
        l = self.x1
        t = self.y1
        for i in range(numRows):
            if i < len(self.gtRows):
                b = self.gtRows[i].y1
            else:
                b = self.y2
            for j in range(numCols):
                if j < len(self.gtCols):
                    r = self.gtCols[j].x1
                else:
                    r = self.x2
                cell = Cell(l, t, r, b, i, j)

                self.gtCells[i][j] = cell
                self.cells.append(cell)

                l = r
            l = self.x1
            t = b

    # public GTElement getSpanAtPosition(int x, int y) {
    # 	ArrayList<GTElement> elems = self.gtSpans
    # 	for(int i=0 i<elems.size() i++)
    # 		if(elems.get(i).contains(x, y))
    # 			return elems.get(i)

    # 	return null
    # }

    # def getElementAtPosition(self, x, y):
    # 	cols = self.gtCols
    # 	for i in range(len(self.gtCols)):
    # 		if self.gtCols[i].contains(x,y)
    # 			return self.gtCols[i]

    # 	rows = self.gtRows
    # 	for(int i=0 i<rows.size() i++)
    # 		if(rows.get(i).contains(x, y))
    # 			return rows.get(i)
    # 	return None

    def get_xml_object(self):
        out_root = ET.Element("GroundTruth")
        out_tables = ET.SubElement(out_root, "Tables")

        out_table = ET.SubElement(out_tables, "Table")
        out_table.attrib["x0"] = str(0)
        out_table.attrib["x1"] = str(self.w)
        out_table.attrib["y0"] = str(0)
        out_table.attrib["y1"] = str(self.h)
        out_table.attrib["orientation"] = "unknown"

        for row in self.gtRows:
            out_row = ET.SubElement(out_table, "Row")

            out_row.attrib["x0"] = str(0)
            out_row.attrib["x1"] = str(self.w)
            out_row.attrib["y0"] = str(row.y1)
            out_row.attrib["y1"] = str(row.y2)

        for col in self.gtCols:
            out_col = ET.SubElement(out_table, "Column")

            out_col.attrib["x0"] = str(col.x1)
            out_col.attrib["x1"] = str(col.x2)
            out_col.attrib["y0"] = str(0)
            out_col.attrib["y1"] = str(self.h)

        # for i in range(len(self.cells)):
        #     for j in range(len(self.cells[i])):
        #         for key in self.cells[i][j].attrib.keys():
        #             self.cells[i][j].attrib[key] = str(self.cells[i][j].attrib[key])
        #         self.cells[i][j].attrib.pop('colspan')
        #         self.cells[i][j].attrib.pop('rowspan')
        #         cell = ET.SubElement(out_table, "Cell")
        #         cell.attrib = self.cells[i][j].attrib
        #         cell.text = self.cells[i][j].text

        self.evaluateCells()
        for i in range(len(self.gtCells)):
            for j in range(len(self.gtCells[i])):
                cell = self.gtCells[i][j]
                out_cell = ET.SubElement(out_table, "Cell")

                out_cell.attrib["x0"] = str(cell.x1)
                out_cell.attrib["x1"] = str(cell.x2)
                out_cell.attrib["y0"] = str(cell.y1)
                out_cell.attrib["y1"] = str(cell.y2)

                out_cell.attrib["startRow"] = str(cell.startRow)
                out_cell.attrib["endRow"] = str(cell.endRow)
                out_cell.attrib["startCol"] = str(cell.startCol)
                out_cell.attrib["endCol"] = str(cell.endCol)

                out_cell.attrib["dontCare"] = str(
                    "true" if cell.dontCare is True else "false"
                )
                out_cell.text = "(0,0,0)"

        return out_root
