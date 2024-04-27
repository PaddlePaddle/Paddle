#pragma once

namespace pir {

class Block;

}

namespace cinn::dialect {

void RewriteCinnOpToPdOp(pir::Block*);

}